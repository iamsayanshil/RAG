"""
Downloads ai4bharat/MSMARCO-XI from the Hugging Face Hub, runs it through the
multi-strategy chunking pipeline, and builds + persists the FAISS + BM25
index to `VECTOR_INDEX_DIR` (see app/config.py).

Usage:
    python scripts/build_index.py [--max-docs N] [--split train]

Run this once before starting the API server (or whenever the dataset /
chunking config changes).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402
from app.utils.logger import logger  # noqa: E402


def load_documents(dataset_name: str, split: str, max_docs: int) -> list[dict]:
    """Loads MSMARCO-XI and normalizes it into {"text", "doc_id", ...} dicts.

    MSMARCO-style datasets typically expose passage text under a `passage`,
    `context`, or nested `passages.passage_text` field depending on the exact
    config used; we probe a few common field names defensively so this
    script keeps working even if the upstream schema shifts slightly.
    """
    from datasets import load_dataset

    logger.info(f'"Downloading {dataset_name} ({split}, max {max_docs} docs)..."')
    try:
        ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        logger.warning(f'"Failed loading {dataset_name}: {e}. Falling back to rajpurkar/squad..."')
        dataset_name = "rajpurkar/squad"
        ds = load_dataset(dataset_name, split="train", streaming=True)

    documents = []
    try:
        for i, row in enumerate(ds):
            if i >= max_docs:
                break
            text = None
            doc_id = row.get("id", i)

            if "passage" in row and isinstance(row["passage"], str):
                text = row["passage"]
            elif "context" in row and isinstance(row["context"], str):
                text = row["context"]
            elif "passages" in row and isinstance(row["passages"], dict):
                texts = row["passages"].get("passage_text") or []
                text = " ".join(texts) if texts else None
            elif "text" in row and isinstance(row["text"], str):
                text = row["text"]

            if not text or not text.strip():
                continue

            documents.append(
                {
                    "text": text.strip(),
                    "doc_id": doc_id,
                    "source": dataset_name,
                    "query": row.get("query"),
                }
            )
    except Exception as err:
        logger.warning(f'"Streaming error on {dataset_name}: {err}. Retrying with rajpurkar/squad non-streaming..."')
        ds_fallback = load_dataset("rajpurkar/squad", split=f"train[:{max_docs}]")
        documents = [
            {"text": row["context"].strip(), "doc_id": i, "source": "rajpurkar/squad"}
            for i, row in enumerate(ds_fallback)
            if row.get("context") and row["context"].strip()
        ]

    logger.info(f'"Loaded {len(documents)} usable documents."')
    return documents


def main():
    parser = argparse.ArgumentParser()
    settings = get_settings()
    parser.add_argument("--dataset", default=settings.DATASET_NAME)
    parser.add_argument("--split", default=settings.DATASET_SPLIT)
    parser.add_argument("--max-docs", type=int, default=settings.DATASET_MAX_DOCS)
    args = parser.parse_args()

    documents = load_documents(args.dataset, args.split, args.max_docs)
    if not documents:
        logger.error('"No documents loaded -- aborting index build."')
        raise SystemExit(1)

    t0 = time.perf_counter()
    retriever = Retriever(settings)
    retriever.build_index(documents)
    retriever.save()
    elapsed = time.perf_counter() - t0

    n_chunks = len(retriever.vector_store.chunks)
    logger.info(
        f'"Index built: {n_chunks} chunks from {len(documents)} docs in {elapsed:.1f}s. '
        f'Saved to {settings.VECTOR_INDEX_DIR}"'
    )


if __name__ == "__main__":
    main()
