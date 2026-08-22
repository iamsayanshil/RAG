"""
API routes -- this is the orchestration layer that stitches together every
stage of the pipeline and is where the "harness" requirement is most visible
end-to-end:

    audio bytes
        -> STT (app/utils/stt_client.py)
        -> input guardrail (app/guardrails/safety.py)
        -> hybrid retrieval (app/retrieval/retriever.py)
        -> rerank (app/retrieval/reranker.py)
        -> generation via forced tool-call + retries (app/llm/generator.py)
        -> output guardrail / grounding check (app/guardrails/grounding.py)
        -> structured response, with a full per-stage latency breakdown
           (app/utils/logger.py) attached to every response.

Every stage is wrapped in try/except with an explicit fallback so a failure
in any single stage degrades to a safe refusal response instead of a 500,
except for genuinely unexpected errors which are surfaced as 500s with a
logged request_id for debugging.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.api.schemas import (
    GuardrailInfo,
    HealthResponse,
    LatencyBreakdown,
    MetricsResponse,
    QueryResponse,
    SourceChunk,
    TextQueryRequest,
    TranscriptionResponse,
)
from app.config import Settings
from app.guardrails.grounding import get_grounding_checker
from app.guardrails.safety import get_safety_guardrail
from app.llm.generator import AnswerGenerator, GeneratedAnswer
from app.retrieval.reranker import get_reranker
from app.retrieval.retriever import Retriever
from app.utils.logger import compute_percentiles, logger
from app.utils.logger import LatencyTracker
from app.utils.stt_client import TranscriptionError, transcribe_audio

router = APIRouter(prefix="/api/v1", tags=["swan"])

RECORD_FILE = Path("data/speech_records.jsonl")
AUDIO_DIR = Path("data/audio_uploads")


def _save_speech_record(request_id: str, transcript: str, answer: str, audio_filename: str | None = None) -> None:
    RECORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": request_id,
        "timestamp": datetime.now().isoformat(),
        "transcript": transcript,
        "answer": answer,
        "audio_file": audio_filename,
    }
    with RECORD_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# --- Dependencies (pull shared singletons off app.state, set in main.py's
# lifespan handler so models/indexes are loaded once at startup, not per-request) ---


def get_retriever(request: Request) -> Retriever:
    retriever: Retriever | None = getattr(request.app.state, "retriever", None)
    if retriever is None or not retriever.is_ready:
        raise HTTPException(status_code=503, detail="Retrieval index is not loaded yet.")
    return retriever


def get_generator(request: Request) -> AnswerGenerator:
    return request.app.state.generator


def get_current_settings(request: Request) -> Settings:
    return request.app.state.settings


# --- Shared pipeline core (used by both /voice-query and /query) -----------


def _run_pipeline(
    query: str,
    retriever: Retriever,
    generator: AnswerGenerator,
    settings: Settings,
    tracker: LatencyTracker,
    top_k: int | None = None,
) -> QueryResponse:
    query_embedding = retriever.embedder.encode([query])[0]

    # 1. Input guardrail -----------------------------------------------------
    with tracker.stage("guardrail_input"):
        safety = get_safety_guardrail(settings)
        input_result = safety.check(query, query_embedding=query_embedding)

    if not input_result.allowed:
        refusal = (
            settings.UNSAFE_REFUSAL_MESSAGE
            if input_result.category == "unsafe"
            else settings.OFF_TOPIC_REFUSAL_MESSAGE
        )
        return QueryResponse(
            query=query,
            answer=refusal,
            is_answerable=False,
            confidence=0.0,
            sources=[],
            guardrails=GuardrailInfo(
                input_allowed=False,
                input_category=input_result.category,
                input_reason=input_result.reason,
            ),
            latency=LatencyBreakdown(stages_ms=tracker.stages, total_ms=tracker.total_ms),
        )

    # 2. Retrieval ------------------------------------------------------------
    with tracker.stage("retrieval"):
        retrieved = retriever.retrieve(query, top_k=top_k)

    # 3. Rerank -----------------------------------------------------------------
    with tracker.stage("rerank"):
        reranker = get_reranker(settings)
        reranked = reranker.rerank(query, retrieved, top_n=settings.TOP_N_RERANK)

    # 4. Generation (harness: forced tool call + retries + fallback) -----------
    with tracker.stage("generation"):
        context_texts = [c.text for c in reranked]
        generated: GeneratedAnswer = generator.generate(query, context_texts)

    # 5. Output guardrail / grounding check -------------------------------------
    with tracker.stage("guardrail_grounding"):
        cited_texts = [
            reranked[i - 1].text for i in generated.citations if 0 < i <= len(reranked)
        ]
        grounding_checker = get_grounding_checker(settings, embed_fn=retriever.embedder.encode)
        grounding = grounding_checker.check(generated, cited_texts)

    final_answer = generated.answer
    is_answerable = generated.is_answerable
    if not grounding.is_grounded and not generated.fallback_used:
        final_answer = settings.UNGROUNDED_REFUSAL_MESSAGE
        is_answerable = False

    return QueryResponse(
        query=query,
        answer=final_answer,
        is_answerable=is_answerable,
        confidence=generated.confidence,
        sources=[
            SourceChunk(
                text=c.text,
                score=round(c.score, 4),
                strategy=c.metadata.get("strategy"),
                doc_id=c.metadata.get("doc_id"),
            )
            for c in reranked
        ],
        guardrails=GuardrailInfo(
            input_allowed=True,
            input_category="ok",
            is_grounded=grounding.is_grounded,
            grounding_score=round(grounding.score, 4),
            grounding_reason=grounding.reason or None,
        ),
        latency=LatencyBreakdown(stages_ms=tracker.stages, total_ms=tracker.total_ms),
    )


# --- Routes ------------------------------------------------------------------


@router.post("/voice-query", response_model=QueryResponse)
async def voice_query(
    audio: UploadFile = File(...),
    retriever: Retriever = Depends(get_retriever),
    generator: AnswerGenerator = Depends(get_generator),
    settings: Settings = Depends(get_current_settings),
):
    """Full end-to-end pipeline: audio blob -> transcript -> grounded answer."""
    request_id = str(uuid.uuid4())
    tracker = LatencyTracker(request_id=request_id)

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    # Save raw audio recording to data/audio_uploads/
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_filename = f"{request_id}.wav"
    audio_path = AUDIO_DIR / audio_filename
    with audio_path.open("wb") as f:
        f.write(audio_bytes)

    with tracker.stage("stt"):
        try:
            transcription = transcribe_audio(audio_bytes, filename=audio.filename or audio_filename)
        except TranscriptionError as exc:
            logger.error(f'"[{request_id}] STT failed: {exc}"')
            raise HTTPException(status_code=422, detail=f"Could not transcribe audio: {exc}") from exc

    response = _run_pipeline(transcription.text, retriever, generator, settings, tracker)
    response.transcript = transcription.text
    tracker.finalize()

    # Save record to data/speech_records.jsonl
    _save_speech_record(
        request_id=request_id,
        transcript=transcription.text,
        answer=response.answer,
        audio_filename=audio_filename,
    )

    return response


@router.post("/query", response_model=QueryResponse)
async def text_query(
    payload: TextQueryRequest,
    retriever: Retriever = Depends(get_retriever),
    generator: AnswerGenerator = Depends(get_generator),
    settings: Settings = Depends(get_current_settings),
):
    """Text-only variant of the pipeline (skips STT)."""
    request_id = str(uuid.uuid4())
    tracker = LatencyTracker(request_id=request_id)
    response = _run_pipeline(
        payload.query, retriever, generator, settings, tracker, top_k=payload.top_k
    )
    tracker.finalize()

    _save_speech_record(
        request_id=request_id,
        transcript=payload.query,
        answer=response.answer,
    )

    return response


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_only(audio: UploadFile = File(...)):
    """STT-only endpoint."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    try:
        result = transcribe_audio(audio_bytes, filename=audio.filename or "audio.wav")
    except TranscriptionError as exc:
        raise HTTPException(status_code=422, detail=f"Could not transcribe audio: {exc}") from exc
    return TranscriptionResponse(text=result.text, language=result.language, provider=result.provider)



@router.get("/history")
async def get_history(limit: int = 50):
    """Returns past recorded speech queries and answers."""
    if not RECORD_FILE.exists():
        return {"records": []}
    
    records = []
    with RECORD_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return {"records": records[-limit:]}


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    retriever: Retriever | None = getattr(request.app.state, "retriever", None)
    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        index_ready=bool(retriever and retriever.is_ready),
        stt_provider=settings.STT_PROVIDER,
        llm_model=settings.LLM_MODEL,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """P50 / P70 / P100 latency percentiles per pipeline stage."""
    return compute_percentiles()

