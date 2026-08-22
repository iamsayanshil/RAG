"""
Central configuration for the S.W.A.N backend.

All tunables live here so latency/behavior can be adjusted without touching
pipeline code. Values are loaded from environment variables / .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    APP_NAME: str = "S.W.A.N Backend"
    ENV: Literal["dev", "prod"] = "dev"
    CORS_ORIGINS: list[str] = ["*"]  # tighten in prod to the deployed frontend origin

    # --- Speech-to-text ---
    # SWAN's frontend just streams a raw audio blob to us; we pick ONE provider here.
    STT_PROVIDER: Literal["elevenlabs", "sarvam"] = "elevenlabs"
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_STT_MODEL: str = "scribe_v2"
    SARVAM_API_KEY: str = ""
    SARVAM_STT_MODEL: str = "saarika:v2"
    STT_LANGUAGE_CODE: str | None = None  # None => auto-detect

    # --- LLM (answer generation) ---
    OPENAI_API_KEY: str = ".env"
    LLM_MODEL: str = "gpt-5.6-luna"
    LLM_MAX_TOKENS: int = 4096
    LLM_TIMEOUT_S: int = 60
    LLM_MAX_RETRIES: int = 3

    # --- Retrieval / vector DB ---
    DATASET_NAME: str = "ai4bharat/MSMARCO-XI"
    DATASET_SPLIT: str = "train"
    DATASET_MAX_DOCS: int = 5000  # cap for demo-scale indexing; raise for full corpus
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    VECTOR_INDEX_DIR: str = "data/index"
    TOP_K_RETRIEVE: int = 10
    TOP_N_RERANK: int = 4
    HYBRID_DENSE_WEIGHT: float = 0.6  # dense vs. BM25 fusion weight

    # --- Chunking (see app/retrieval/chunking.py for the strategies themselves) ---
    CHUNK_STRATEGIES: list[str] = ["fixed", "sliding_window", "sentence_semantic", "metadata_aware"]
    FIXED_CHUNK_SIZE_TOKENS: int = 128
    FIXED_CHUNK_OVERLAP_TOKENS: int = 0
    SLIDING_CHUNK_SIZE_TOKENS: int = 96
    SLIDING_CHUNK_OVERLAP_TOKENS: int = 32
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.55
    SEMANTIC_MAX_SENTENCES: int = 8

    # --- Guardrails ---
    ENABLE_SAFETY_GUARDRAIL: bool = True
    ENABLE_GROUNDING_GUARDRAIL: bool = True
    GROUNDING_MIN_SCORE: float = 0.45
    OFF_TOPIC_MIN_SCORE: float = 0.28
    UNSAFE_REFUSAL_MESSAGE: str = (
        "I can't help with that request. Please ask something related to the knowledge base."
    )
    OFF_TOPIC_REFUSAL_MESSAGE: str = (
        "That looks outside what I can answer from the current knowledge base. "
        "Try rephrasing or asking something covered by the indexed documents."
    )
    UNGROUNDED_REFUSAL_MESSAGE: str = (
        "I found some related passages but couldn't confidently ground an answer in them, "
        "so I'd rather not guess. Could you rephrase or narrow the question?"
    )

    # --- Latency ---
    LATENCY_BUDGET_MS: int = 200  # target for chunking+retrieval+rerank+guardrails+generation
    LATENCY_LOG_PATH: str = "data/latency_log.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
