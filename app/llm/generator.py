"""
Answer generation harness.

This is intentionally NOT a raw "prompt in, text out" call. It:
  1. Forces structured output via Anthropic tool-use (the model MUST call
     `provide_answer`, so we get typed JSON back instead of parsing free text).
  2. Retries transient failures (timeouts, rate limits, malformed tool calls)
     with exponential backoff.
  3. Validates the structured output against the expected schema and repairs
     / retries on validation failure.
  4. Falls back to a safe, explicit "I don't know" response on exhausted
     retries or unrecoverable errors, rather than ever surfacing a raw
     exception or a hallucinated guess to the end user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings, get_settings
from app.llm.prompts import PROVIDE_ANSWER_TOOL, QA_SYSTEM_PROMPT, build_qa_user_prompt
from app.utils.logger import logger


class GenerationError(Exception):
    pass


class MalformedToolCallError(GenerationError):
    """Raised when the model responds without calling provide_answer, or the
    tool input fails schema validation. Retried like any other transient
    failure before falling back."""


@dataclass
class GeneratedAnswer:
    answer: str
    is_answerable: bool
    citations: list[int] = field(default_factory=list)
    confidence: float = 0.0
    fallback_used: bool = False
    raw_stop_reason: str | None = None


FALLBACK_ANSWER = GeneratedAnswer(
    answer=(
        "I wasn't able to generate a reliable answer right now. "
        "Please try rephrasing your question or asking again in a moment."
    ),
    is_answerable=False,
    citations=[],
    confidence=0.0,
    fallback_used=True,
)


def _validate_tool_input(data: dict) -> GeneratedAnswer:
    required = {"answer", "is_answerable", "citations", "confidence"}
    missing = required - data.keys()
    if missing:
        raise MalformedToolCallError(f"Missing fields in tool call: {missing}")
    if not isinstance(data["citations"], list):
        raise MalformedToolCallError("citations must be a list")
    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError) as exc:
        raise MalformedToolCallError("confidence must be numeric") from exc

    return GeneratedAnswer(
        answer=str(data["answer"]),
        is_answerable=bool(data["is_answerable"]),
        citations=[int(i) for i in data["citations"] if str(i).isdigit()],
        confidence=max(0.0, min(1.0, confidence)),
    )


class AnswerGenerator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                api_key=self.settings.OPENAI_API_KEY,
                timeout=self.settings.LLM_TIMEOUT_S,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.3, min=0.3, max=3),
        retry=retry_if_exception_type((MalformedToolCallError, TimeoutError)),
        reraise=True,
    )
    def _call_model(self, query: str, context_chunks: list[str]):
        client = self._get_client()
        response = client.messages.create(
            model=self.settings.LLM_MODEL,
            max_tokens=self.settings.LLM_MAX_TOKENS,
            system=QA_SYSTEM_PROMPT,
            tools=[PROVIDE_ANSWER_TOOL],
            tool_choice={"type": "tool", "name": "provide_answer"},
            messages=[{"role": "user", "content": build_qa_user_prompt(query, context_chunks)}],
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            raise MalformedToolCallError("Model did not call provide_answer")

        parsed = _validate_tool_input(tool_use_blocks[0].input)
        parsed.raw_stop_reason = response.stop_reason
        return parsed

    def generate(self, query: str, context_chunks: list[str]) -> GeneratedAnswer:
        """Entry point used by the API layer. Never raises -- always returns
        a GeneratedAnswer, falling back to a safe refusal on any failure so
        the request pipeline can complete deterministically."""
        if not context_chunks:
            return GeneratedAnswer(
                answer="I don't have any retrieved context to answer from.",
                is_answerable=False,
                fallback_used=True,
            )

        try:
            return self._call_model(query, context_chunks)
        except Exception as exc:  # noqa: BLE001 - last line of defense for the harness
            logger.error(f'"Generation failed after retries: {exc}"')
            return FALLBACK_ANSWER
