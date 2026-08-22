"""
Reranker: re-scores the top-K hybrid retrieval hits with a cross-encoder for
higher precision before they're handed to the LLM. Cross-encoders see the
query and passage together (rather than as separate embeddings), which is
slower but meaningfully more accurate at judging true relevance -- so we only
ever run it over the small top-K shortlist, not the whole corpus.

Falls back to a cheap lexical-overlap score if the cross-encoder can't be
loaded in time (e.g. cold start under latency pressure), so the pipeline
degrades gracefully instead of failing.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.retrieval.retriever import RetrievedChunk
from app.utils.logger import logger


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._load_failed = False

    def _load(self):
        if self._model is None and not self._load_failed:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f'"Loading cross-encoder reranker {self.model_name}"')
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:  # noqa: BLE001
                logger.error(f'"Cross-encoder load failed, falling back to lexical rerank: {exc}"')
                self._load_failed = True
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        model = self._load()
        if model is None:
            return _lexical_rerank(query, candidates, top_n)

        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.score = float(s)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_n]


def _lexical_rerank(
    query: str, candidates: list[RetrievedChunk], top_n: int
) -> list[RetrievedChunk]:
    """Jaccard token-overlap fallback -- no model load required."""
    q_tokens = set(query.lower().split())
    for c in candidates:
        c_tokens = set(c.text.lower().split())
        if not c_tokens:
            overlap = 0.0
        else:
            overlap = len(q_tokens & c_tokens) / len(q_tokens | c_tokens)
        c.score = overlap
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_n]


_reranker_singleton: CrossEncoderReranker | None = None


def get_reranker(settings: Settings | None = None) -> CrossEncoderReranker:
    global _reranker_singleton
    if _reranker_singleton is None:
        settings = settings or get_settings()
        _reranker_singleton = CrossEncoderReranker(settings.CROSS_ENCODER_MODEL_NAME)
    return _reranker_singleton
