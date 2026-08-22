"""Pydantic request/response models for the API layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    text: str
    score: float
    strategy: str | None = None
    doc_id: str | int | None = None


class GuardrailInfo(BaseModel):
    input_allowed: bool
    input_category: str  # "ok" | "unsafe" | "off_topic"
    input_reason: str | None = None
    is_grounded: bool | None = None
    grounding_score: float | None = None
    grounding_reason: str | None = None


class LatencyBreakdown(BaseModel):
    stages_ms: dict[str, float]
    total_ms: float


class QueryResponse(BaseModel):
    transcript: str | None = None  # populated only for /voice-query
    query: str
    answer: str
    is_answerable: bool
    confidence: float
    sources: list[SourceChunk] = Field(default_factory=list)
    guardrails: GuardrailInfo
    latency: LatencyBreakdown


class TextQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class TranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    provider: str


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    stt_provider: str
    llm_model: str


class StagePercentiles(BaseModel):
    p50: float
    p70: float
    p100: float
    mean: float


class MetricsResponse(BaseModel):
    count: int
    stages: dict[str, StagePercentiles]
    total: StagePercentiles | None = None
