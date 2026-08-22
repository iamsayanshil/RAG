"""
Input-side guardrail: runs BEFORE retrieval/generation.

Two checks, both fast (no LLM call, so they don't eat into the latency
budget):
  1. Unsafe/inappropriate content -- regex/keyword blocklist for clearly
     disallowed categories (self-harm facilitation, explicit violence
     instructions, etc). This is a coarse first line of defense, not a
     substitute for provider-level moderation in a real deployment.
  2. Off-topic detection -- embedding similarity between the query and a
     "domain centroid" (the mean embedding of a sample of indexed corpus
     chunks). Queries that don't resemble the corpus at all (e.g. "write me
     a poem about dragons" against an MS MARCO passage corpus) are flagged
     rather than sent through retrieval + generation to produce a
     low-quality or made-up answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from app.config import Settings, get_settings

_UNSAFE_PATTERNS = [
    r"\bhow (?:do|can) i (?:make|build|synthesi[sz]e) (?:a )?(?:bomb|explosive|weapon)\b",
    r"\bhow (?:do|can) i (?:kill|harm|hurt) (?:myself|someone|him|her|them)\b",
    r"\bself[- ]harm\b",
    r"\bchild (?:sexual|abuse)\b",
    r"\bhow (?:do|can) i (?:hack|exploit) .* (?:without (?:permission|authoriz)|illegally)\b",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)


@dataclass
class GuardrailResult:
    allowed: bool
    category: str  # "ok" | "unsafe" | "off_topic"
    reason: str
    score: float | None = None


class SafetyGuardrail:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._domain_centroid: np.ndarray | None = None

    def set_domain_centroid(self, centroid: np.ndarray) -> None:
        """Called once at startup with the mean embedding of a corpus sample."""
        self._domain_centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    def check_unsafe(self, text: str) -> GuardrailResult:
        if _UNSAFE_RE.search(text):
            return GuardrailResult(
                allowed=False,
                category="unsafe",
                reason="Query matched a disallowed content pattern.",
            )
        return GuardrailResult(allowed=True, category="ok", reason="")

    def check_off_topic(self, query_embedding: np.ndarray) -> GuardrailResult:
        if self._domain_centroid is None:
            # No centroid computed yet (e.g. empty index) -- don't block on it.
            return GuardrailResult(allowed=True, category="ok", reason="", score=None)

        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        similarity = float(np.dot(q, self._domain_centroid))

        if similarity < self.settings.OFF_TOPIC_MIN_SCORE:
            return GuardrailResult(
                allowed=False,
                category="off_topic",
                reason=f"Query similarity to corpus domain ({similarity:.2f}) below threshold.",
                score=similarity,
            )
        return GuardrailResult(allowed=True, category="ok", reason="", score=similarity)

    def check(self, text: str, query_embedding: np.ndarray | None = None) -> GuardrailResult:
        """Runs all enabled input-side checks; returns the first failure, or
        an "ok" result if everything passes."""
        if not self.settings.ENABLE_SAFETY_GUARDRAIL:
            return GuardrailResult(allowed=True, category="ok", reason="disabled")

        unsafe_result = self.check_unsafe(text)
        if not unsafe_result.allowed:
            return unsafe_result

        if query_embedding is not None:
            off_topic_result = self.check_off_topic(query_embedding)
            if not off_topic_result.allowed:
                return off_topic_result

        return GuardrailResult(allowed=True, category="ok", reason="")


_singleton: SafetyGuardrail | None = None


def get_safety_guardrail(settings: Settings | None = None) -> SafetyGuardrail:
    global _singleton
    if _singleton is None:
        _singleton = SafetyGuardrail(settings or get_settings())
    return _singleton
