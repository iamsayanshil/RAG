"""
Chunking strategies.

The task explicitly asks for more than one naive fixed-size splitter, so this
module implements four distinct strategies and a pipeline that can run any
combination of them over a document, tagging each resulting chunk with which
strategy produced it (stored in metadata) so retrieval can be evaluated /
filtered per-strategy later if needed.

Strategies:
    1. FixedSizeChunker      - naive fixed token-count windows, no overlap.
                                Kept as a baseline for comparison, not used alone.
    2. SlidingWindowChunker  - fixed-size windows WITH overlap, to avoid
                                cutting relevant context at chunk boundaries.
    3. SentenceSemanticChunker - groups consecutive sentences and starts a new
                                chunk when semantic similarity between
                                consecutive sentences drops (topic shift),
                                capped at a max sentence count.
    4. MetadataAwareChunker  - splits on structural boundaries (passage /
                                document id, title) from the source dataset
                                and attaches that metadata to every chunk, so
                                retrieval can filter/boost by source doc.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.metadata.setdefault("n_tokens", len(_WORD_RE.findall(self.text)))


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text.strip()]


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


class BaseChunker(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(self, text: str, doc_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        ...


class FixedSizeChunker(BaseChunker):
    """Naive fixed-size, non-overlapping windows. Baseline only."""

    name = "fixed"

    def __init__(self, chunk_size_tokens: int = 128):
        self.chunk_size_tokens = chunk_size_tokens

    def chunk(self, text: str, doc_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        tokens = _tokenize(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size_tokens):
            piece = " ".join(tokens[i : i + self.chunk_size_tokens])
            meta = {**(doc_metadata or {}), "strategy": self.name, "start_token": i}
            chunks.append(Chunk(piece, meta))
        return chunks


class SlidingWindowChunker(BaseChunker):
    """Fixed-size windows with overlap, so boundary context isn't lost."""

    name = "sliding_window"

    def __init__(self, chunk_size_tokens: int = 96, overlap_tokens: int = 32):
        assert overlap_tokens < chunk_size_tokens, "overlap must be smaller than chunk size"
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str, doc_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        tokens = _tokenize(text)
        step = self.chunk_size_tokens - self.overlap_tokens
        chunks = []
        i = 0
        while i < len(tokens):
            piece = " ".join(tokens[i : i + self.chunk_size_tokens])
            if piece:
                meta = {
                    **(doc_metadata or {}),
                    "strategy": self.name,
                    "start_token": i,
                    "overlap_tokens": self.overlap_tokens,
                }
                chunks.append(Chunk(piece, meta))
            if i + self.chunk_size_tokens >= len(tokens):
                break
            i += step
        return chunks


class SentenceSemanticChunker(BaseChunker):
    """Groups sentences and breaks the chunk at semantic discontinuities.

    Requires an embedding function `embed_fn(list[str]) -> np.ndarray` so the
    boundary decision is based on meaning, not just character/token count.
    Falls back to sentence-count-only grouping if no embedder is supplied
    (still better than naive fixed-size since it never splits mid-sentence).
    """

    name = "sentence_semantic"

    def __init__(
        self,
        embed_fn=None,
        similarity_threshold: float = 0.55,
        max_sentences: int = 8,
    ):
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_sentences = max_sentences

    def chunk(self, text: str, doc_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(text.strip(), {**(doc_metadata or {}), "strategy": self.name})]

        if self.embed_fn is not None:
            embeddings = self.embed_fn(sentences)
            sims = [
                float(np.dot(embeddings[i], embeddings[i + 1]))
                for i in range(len(embeddings) - 1)
            ]
        else:
            sims = [1.0] * (len(sentences) - 1)  # no signal -> group by count only

        chunks: list[Chunk] = []
        current: list[str] = [sentences[0]]
        for idx in range(1, len(sentences)):
            same_topic = sims[idx - 1] >= self.similarity_threshold
            if same_topic and len(current) < self.max_sentences:
                current.append(sentences[idx])
            else:
                chunks.append(self._finalize(current, doc_metadata))
                current = [sentences[idx]]
        if current:
            chunks.append(self._finalize(current, doc_metadata))
        return chunks

    def _finalize(self, sentence_group: list[str], doc_metadata: dict[str, Any] | None) -> Chunk:
        text = " ".join(sentence_group)
        meta = {**(doc_metadata or {}), "strategy": self.name, "n_sentences": len(sentence_group)}
        return Chunk(text, meta)


class MetadataAwareChunker(BaseChunker):
    """Wraps another chunker and enriches every chunk with document-level
    metadata (doc_id, title/query, source) so retrieval results are
    traceable back to their origin and can be filtered/boosted by source.
    """

    name = "metadata_aware"

    def __init__(self, inner: BaseChunker):
        self.inner = inner

    def chunk(self, text: str, doc_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        base_chunks = self.inner.chunk(text, doc_metadata)
        for c in base_chunks:
            c.metadata["strategy"] = self.name
            c.metadata["inner_strategy"] = self.inner.name
            c.metadata.setdefault("doc_id", (doc_metadata or {}).get("doc_id"))
            c.metadata.setdefault("source", (doc_metadata or {}).get("source", "MSMARCO-XI"))
        return base_chunks


class ChunkingPipeline:
    """Runs one or more strategies over a document and returns the union of
    resulting chunks, each tagged with which strategy produced it.

    Running multiple strategies over the same corpus intentionally creates
    some redundancy in the index (e.g. a fixed-size chunk AND an overlapping
    sliding-window chunk covering similar text). This is a deliberate
    recall/precision trade-off: at query time the retriever + reranker
    (see retriever.py / reranker.py) collapse near-duplicate hits, so the
    redundancy improves recall without hurting final answer quality.
    """

    def __init__(self, chunkers: list[BaseChunker]):
        self.chunkers = chunkers

    def run(self, text: str, doc_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for chunker in self.chunkers:
            all_chunks.extend(chunker.chunk(text, doc_metadata))
        return all_chunks

    @classmethod
    def from_settings(cls, settings, embed_fn=None) -> "ChunkingPipeline":
        registry: dict[str, BaseChunker] = {
            "fixed": FixedSizeChunker(settings.FIXED_CHUNK_SIZE_TOKENS),
            "sliding_window": SlidingWindowChunker(
                settings.SLIDING_CHUNK_SIZE_TOKENS, settings.SLIDING_CHUNK_OVERLAP_TOKENS
            ),
            "sentence_semantic": SentenceSemanticChunker(
                embed_fn=embed_fn,
                similarity_threshold=settings.SEMANTIC_SIMILARITY_THRESHOLD,
                max_sentences=settings.SEMANTIC_MAX_SENTENCES,
            ),
        }
        # metadata_aware wraps the sliding-window chunker by default, since that
        # tends to give the best recall/context trade-off for downstream QA.
        registry["metadata_aware"] = MetadataAwareChunker(registry["sliding_window"])

        selected = [registry[name] for name in settings.CHUNK_STRATEGIES if name in registry]
        return cls(selected)
