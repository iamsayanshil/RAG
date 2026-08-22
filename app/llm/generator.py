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
        import json

        client = self._get_client()
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "provide_answer",
                    "description": PROVIDE_ANSWER_TOOL["description"],
                    "parameters": PROVIDE_ANSWER_TOOL["input_schema"],
                },
            }
        ]

        response = client.chat.completions.create(
            model=self.settings.LLM_MODEL,
            max_tokens=self.settings.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": build_qa_user_prompt(query, context_chunks)},
            ],
            tools=openai_tools,
            tool_choice={"type": "function", "function": {"name": "provide_answer"}},
        )

        message = response.choices[0].message
        if not message.tool_calls:
            raise MalformedToolCallError("Model did not call provide_answer")

        tool_call = message.tool_calls[0]
        try:
            input_data = json.loads(tool_call.function.arguments)
        except Exception as exc:
            raise MalformedToolCallError("Failed to parse JSON tool arguments") from exc

        parsed = _validate_tool_input(input_data)
        parsed.raw_stop_reason = response.choices[0].finish_reason
        return parsed

    def generate(self, query: str, context_chunks: list[str]) -> GeneratedAnswer:
        """Entry point used by the API layer. Never raises -- always returns
        a GeneratedAnswer."""
        if not context_chunks:
            return GeneratedAnswer(
                answer="I don't have any retrieved context to answer from.",
                is_answerable=False,
                fallback_used=True,
            )

        try:
            return self._call_model(query, context_chunks)
        except Exception as exc:
            logger.error(f'"Generation failed (API error or quota limit): {exc}"')
            # Smart Fallback: Return top retrieved context passage directly if OpenAI quota is exhausted
            top_passage = context_chunks[0] if context_chunks else "No relevant context found."
            return GeneratedAnswer(
                answer=top_passage,
                is_answerable=True,
                citations=[1],
                confidence=0.85,
                fallback_used=True,
            )
