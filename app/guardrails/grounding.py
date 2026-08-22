"""
Output-side guardrail: runs AFTER generation, before the answer is returned
to the user. Checks whether the generated answer is actually supported by
the retrieved context, so the system can withhold/soften an answer instead
of confidently returning a hallucination.

Two independent signals are combined:
  1. The model's OWN self-reported `is_answerable` / `confidence` from the
     structured tool call (see app/llm/generator.py) -- cheap, already
     computed, no extra cost.
  2. An independent embedding-similarity check between the generated answer
     and the cited context chunks -- catches cases where the model claims
     confidence but the answer text doesn't actually overlap with the
     context it says it used (a common hallucination pattern).

This is a heuristic, not a proof of factual correctness -- it's designed to
catch the common failure modes cheaply within the latency budget, not to be
a perfect entailment classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import Settings, get_settings
from app.llm.generator import GeneratedAnswer


@dataclass
class GroundingResult:
    is_grounded: bool
    score: float
    reason: str


class GroundingChecker:
    def __init__(self, settings: Settings | None = None, embed_fn=None):
        self.settings = settings or get_settings()
        self.embed_fn = embed_fn  # injected: app.retrieval.retriever.EmbeddingModel.encode

    def check(
        self,
        generated: GeneratedAnswer,
        cited_context_texts: list[str],
    ) -> GroundingResult:
        if not self.settings.ENABLE_GROUNDING_GUARDRAIL:
            return GroundingResult(is_grounded=True, score=1.0, reason="disabled")

        if not generated.is_answerable:
            return GroundingResult(
                is_grounded=False, score=0.0, reason="Model self-reported unanswerable."
            )

        if not generated.citations or not cited_context_texts:
            return GroundingResult(
                is_grounded=False,
                score=0.0,
                reason="Answer provided no citations into the retrieved context.",
            )

        similarity_score = self._embedding_overlap(generated.answer, cited_context_texts)
        # Blend the model's own confidence with the independent similarity check
        # so a single miscalibrated signal can't pass/fail the answer alone.
        combined = 0.5 * generated.confidence + 0.5 * similarity_score

        if combined < self.settings.GROUNDING_MIN_SCORE:
            return GroundingResult(
                is_grounded=False,
                score=combined,
                reason=(
                    f"Combined grounding score {combined:.2f} below threshold "
                    f"{self.settings.GROUNDING_MIN_SCORE}."
                ),
            )
        return GroundingResult(is_grounded=True, score=combined, reason="")

    def _embedding_overlap(self, answer: str, context_texts: list[str]) -> float:
        if self.embed_fn is None:
            # No embedder injected (e.g. unit test) -- neutral score, rely on
            # the model's self-reported confidence alone.
            return 0.5
        vectors = self.embed_fn([answer, *context_texts])
        answer_vec, context_vecs = vectors[0], vectors[1:]
        sims = context_vecs @ answer_vec / (
            np.linalg.norm(context_vecs, axis=1) * np.linalg.norm(answer_vec) + 1e-9
        )
        return float(np.max(sims)) if len(sims) else 0.0


_singleton: GroundingChecker | None = None


def get_grounding_checker(settings: Settings | None = None, embed_fn=None) -> GroundingChecker:
    global _singleton
    if _singleton is None:
        _singleton = GroundingChecker(settings or get_settings(), embed_fn=embed_fn)
    elif embed_fn is not None and _singleton.embed_fn is None:
        _singleton.embed_fn = embed_fn
    return _singleton
