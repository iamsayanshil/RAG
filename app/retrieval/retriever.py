"""
Retriever: embeddings + FAISS dense index + BM25 sparse index, fused via
weighted reciprocal-rank-ish score combination (hybrid search).

Design notes for the <200ms budget:
- Embedding model is a small local sentence-transformers model (all-MiniLM-L6-v2,
  ~80MB) run on CPU with no network round trip.
- FAISS IndexFlatIP is used for exact search; for the demo-scale corpus
  (a few thousand chunks) this is faster than an approximate index and avoids
  tuning HNSW params under time pressure. Swap to IndexHNSWFlat if the corpus
  grows past ~100k chunks.
- BM25 (rank_bm25) runs in parallel over the same chunk set for lexical
  recall (catches exact keyword/entity matches dense embeddings can miss).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.retrieval.chunking import Chunk, ChunkingPipeline
from app.utils.logger import logger


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_score: float | None = None
    sparse_score: float | None = None


class EmbeddingModel:
    """Thin wrapper around sentence-transformers so it's a single, lazily
    loaded, mockable dependency."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f'"Loading embedding model {self.model_name}"')
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        model = self._load()
        embeddings = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=normalize, show_progress_bar=False
        )
        return embeddings.astype("float32")


class VectorStore:
    """FAISS IndexFlatIP wrapper (cosine similarity via normalized vectors)."""

    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        assert embeddings.shape[0] == len(chunks)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        scores, indices = self.index.search(query_vec.reshape(1, -1), top_k)
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        with (directory / "chunks.pkl").open("wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        import faiss

        index = faiss.read_index(str(directory / "faiss.index"))
        store = cls.__new__(cls)
        store.index = index
        store.dim = index.d
        with (directory / "chunks.pkl").open("rb") as f:
            store.chunks = pickle.load(f)
        return store


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


def _minmax_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class Retriever:
    """Owns the embedding model + dense/sparse indexes and exposes
    `retrieve(query, top_k)` as the single entry point used by the API layer.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embedder = EmbeddingModel(self.settings.EMBEDDING_MODEL_NAME)
        self.vector_store: VectorStore | None = None
        self.bm25: BM25Index | None = None

    # ---- index lifecycle -------------------------------------------------

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """documents: list of {"text": ..., "doc_id": ..., ...metadata}

        Runs the full multi-strategy chunking pipeline (see chunking.py) over
        every document, embeds every resulting chunk, and builds both the
        dense (FAISS) and sparse (BM25) indexes over the union of chunks.
        """
        pipeline = ChunkingPipeline.from_settings(self.settings, embed_fn=self._embed_sentences)

        all_chunks: list[Chunk] = []
        for doc in documents:
            doc_meta = {k: v for k, v in doc.items() if k != "text"}
            all_chunks.extend(pipeline.run(doc["text"], doc_meta))

        logger.info(f'"Chunked {len(documents)} documents into {len(all_chunks)} chunks"')

        embeddings = self.embedder.encode([c.text for c in all_chunks])
        self.vector_store = VectorStore(dim=embeddings.shape[1])
        self.vector_store.add(embeddings, all_chunks)
        self.bm25 = BM25Index(all_chunks)

    def _embed_sentences(self, sentences: list[str]) -> np.ndarray:
        return self.embedder.encode(sentences)

    def save(self) -> None:
        if self.vector_store is None:
            raise RuntimeError("No index built yet.")
        self.vector_store.save(Path(self.settings.VECTOR_INDEX_DIR))

    def load(self) -> bool:
        directory = Path(self.settings.VECTOR_INDEX_DIR)
        if not (directory / "faiss.index").exists():
            return False
        self.vector_store = VectorStore.load(directory)
        self.bm25 = BM25Index(self.vector_store.chunks)
        logger.info(f'"Loaded index with {len(self.vector_store.chunks)} chunks"')
        return True

    @property
    def is_ready(self) -> bool:
        return self.vector_store is not None and self.bm25 is not None

    # ---- query path --------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not self.is_ready:
            raise RuntimeError("Retriever index not loaded. Call load()/build_index() first.")

        top_k = top_k or self.settings.TOP_K_RETRIEVE
        query_vec = self.embedder.encode([query])[0]

        dense_hits = self.vector_store.search(query_vec, top_k)
        sparse_hits = self.bm25.search(query, top_k)

        dense_scores = _minmax_normalize({idx: score for idx, score in dense_hits})
        sparse_scores = _minmax_normalize({idx: score for idx, score in sparse_hits})

        combined_ids = set(dense_scores) | set(sparse_scores)
        w = self.settings.HYBRID_DENSE_WEIGHT

        fused: list[RetrievedChunk] = []
        for idx in combined_ids:
            d = dense_scores.get(idx, 0.0)
            s = sparse_scores.get(idx, 0.0)
            fused_score = w * d + (1 - w) * s
            chunk = self.vector_store.chunks[idx]
            fused.append(
                RetrievedChunk(
                    text=chunk.text,
                    score=fused_score,
                    metadata=chunk.metadata,
                    dense_score=dense_scores.get(idx),
                    sparse_score=sparse_scores.get(idx),
                )
            )

        fused.sort(key=lambda c: c.score, reverse=True)
        return fused[:top_k]