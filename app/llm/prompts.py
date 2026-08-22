"""Prompt templates used by app/llm/generator.py and app/guardrails/*.

Kept separate from generator.py so prompt iteration doesn't require touching
orchestration/harness code.
"""

QA_SYSTEM_PROMPT = """You are S.W.A.N (Spoken Word Analysis Network), a voice-driven \
retrieval-augmented assistant. You answer ONLY using the provided CONTEXT passages.

Rules:
- If the context does not contain enough information to answer confidently, \
say so explicitly instead of guessing.
- Never invent facts, sources, or numbers that are not present in the context.
- Cite which context passage(s) support each claim using their [index] number.
- Keep answers concise and speakable (this will be read aloud / read quickly by a user \
who just asked a question by voice).

You MUST respond by calling the `provide_answer` tool -- never respond with plain text.
"""


def build_qa_user_prompt(query: str, context_chunks: list[str]) -> str:
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{query}\n\n"
        "Call provide_answer with your response."
    )


PROVIDE_ANSWER_TOOL = {
    "name": "provide_answer",
    "description": "Return the final answer to the user's spoken question, grounded in the given context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to speak/display back to the user. Empty string if unanswerable.",
            },
            "is_answerable": {
                "type": "boolean",
                "description": "False if the context does not sufficiently support an answer.",
            },
            "citations": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Indices (1-based) of the context passages actually used.",
            },
            "confidence": {
                "type": "number",
                "description": "Model's own confidence in the answer, from 0.0 to 1.0.",
            },
        },
        "required": ["answer", "is_answerable", "citations", "confidence"],
    },
}


# --- Guardrail prompts (used only as an optional deep-check; the fast path in
# app/guardrails/safety.py and grounding.py is heuristic/embedding based so it
# doesn't add an extra network round trip on every request). ---

OFF_TOPIC_CLASSIFIER_PROMPT = """Decide if the QUESTION below is answerable from a general \
open-domain web-passage knowledge base (MS MARCO-style), or if it is clearly off-topic \
(e.g. asks the assistant to role-play, asks about itself, or requests something unrelated \
to factual lookup). Respond by calling the `classify_topic` tool only."""

CLASSIFY_TOPIC_TOOL = {
    "name": "classify_topic",
    "description": "Classify whether a question is in-domain for the knowledge base.",
    "input_schema": {
        "type": "object",
        "properties": {
            "in_domain": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["in_domain", "reason"],
    },
}
