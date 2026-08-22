"""
FastAPI application entrypoint.

Wires the S.W.A.N frontend (index.html / script.js, which records mic audio
in-browser and is expected to POST it to /api/v1/voice-query) to the backend
pipeline: STT -> guardrails -> hybrid retrieval -> rerank -> generation ->
grounding guardrail.

Run with:
    uvicorn app.main:app --reload --port 8000

The vector index is loaded once at startup (see `lifespan`) so per-request
latency only pays for embedding + search, not model/index loading. If no
index exists on disk yet, run `python scripts/build_index.py` first (see
README.md).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.llm.generator import AnswerGenerator
from app.retrieval.retriever import Retriever
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    logger.info('"Starting S.W.A.N backend..."')

    retriever = Retriever(settings)
    loaded = retriever.load()
    if not loaded:
        logger.error(
            '"No vector index found on disk -- /api/v1/query and /api/v1/voice-query '
            'will return 503 until scripts/build_index.py has been run."'
        )
    else:
        # Warm up the safety guardrail's off-topic detector with a domain
        # centroid computed from a sample of the indexed corpus.
        from app.guardrails.safety import get_safety_guardrail
        import numpy as np

        sample = retriever.vector_store.chunks[:200]
        if sample:
            sample_embeddings = retriever.embedder.encode([c.text for c in sample])
            centroid = np.mean(sample_embeddings, axis=0)
            get_safety_guardrail(settings).set_domain_centroid(centroid)

    from app.retrieval.reranker import get_reranker
    get_reranker(settings)._load()

    app.state.retriever = retriever
    app.state.generator = AnswerGenerator(settings)

    logger.info('"S.W.A.N backend ready."')
    yield
    logger.info('"Shutting down S.W.A.N backend."')


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        description="Voice-enabled RAG backend for S.W.A.N (Spoken Word Analysis Network).",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request, exc):  # noqa: ANN001
        logger.error(f'"Unhandled exception on {request.url.path}: {exc}"')
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. This has been logged."},
        )

    @app.get("/")
    async def root():
        return {"service": settings.APP_NAME, "status": "ok", "docs": "/docs"}

    return app


app = create_app()
