# S.W.A.N Backend — Voice-Enabled RAG Pipeline

Backend for **S.W.A.N (Spoken Word Analysis Network)**. The frontend (`index.html` /
`script.js` / `style.css`) captures microphone audio in-browser via
`getUserMedia` and toggles a "listening" UI state — it does not transcribe or
answer anything itself. This backend is what the recorded audio blob should be
POSTed to: it transcribes it, retrieves grounded context from the MSMARCO-XI
dataset, generates an answer, and returns it.

```
Voice input (browser mic, script.js)
        │  POST /api/v1/voice-query (multipart audio)
        ▼
  Speech-to-text  ──────────────  app/utils/stt_client.py   (ElevenLabs Scribe / Sarvam)
        ▼
  Input guardrail  ─────────────  app/guardrails/safety.py  (unsafe + off-topic check)
        ▼
  Chunking (offline, index-build time)  ── app/retrieval/chunking.py
  Hybrid retrieval (dense + BM25)  ─────  app/retrieval/retriever.py
        ▼
  Rerank (cross-encoder)  ──────────────  app/retrieval/reranker.py
        ▼
  Answer generation (harness)  ─────────  app/llm/generator.py + prompts.py
        ▼
  Output guardrail (grounding check)  ──  app/guardrails/grounding.py
        ▼
  Structured JSON response + latency breakdown
```

## Project structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI app, CORS, startup index/model loading
│   ├── config.py                # * all settings/env vars in one place
│   ├── api/
│   │   ├── routes.py            # orchestrates the full pipeline end-to-end
│   │   └── schemas.py           # request/response models
│   ├── retrieval/
│   │   ├── chunking.py          # * 4 chunking strategies + pipeline
│   │   ├── retriever.py         # embeddings + FAISS + BM25 hybrid search
│   │   └── reranker.py          # cross-encoder reranking
│   ├── llm/
│   │   ├── generator.py         # generation harness (tool-call, retries, fallback)
│   │   └── prompts.py           # system/user prompt + tool schema
│   ├── guardrails/
│   │   ├── safety.py            # input: unsafe content + off-topic detection
│   │   └── grounding.py         # output: hallucination / grounding check
│   └── utils/
│       ├── logger.py            # structured logging + LatencyTracker (P50/P70/P100)
│       └── stt_client.py        # * ElevenLabs / Sarvam STT behind one interface
├── scripts/
│   ├── build_index.py           # * downloads MSMARCO-XI, builds the vector index
│   └── benchmark_latency.py     # * fires N queries, reports latency percentiles
├── data/                        # vector index + latency log get written here
├── requirements.txt
├── .env.example
└── README.md
```

`*` marks files/folders added beyond the originally sketched tree
(`config.py`, `chunking.py`, `stt_client.py`, `scripts/`) — they were needed to
actually satisfy the task requirements (multiple chunking strategies, a
concrete STT integration, a way to build the index and measure latency) and
are placed in the most natural existing folder rather than inventing new
top-level ones.

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in ANTHROPIC_API_KEY and ELEVENLABS_API_KEY (or SARVAM_API_KEY + STT_PROVIDER=sarvam)

python scripts/build_index.py --max-docs 5000   # downloads MSMARCO-XI, builds data/index/
uvicorn app.main:app --reload --port 8000
```

Point `script.js` at `http://localhost:8000/api/v1/voice-query` (record the
`MediaStream` to a `Blob` via `MediaRecorder` and `POST` it as
`multipart/form-data` under the `audio` field) to wire the existing frontend
up end to end.

## Endpoints

| Method | Path                   | Purpose                                              |
|--------|------------------------|-------------------------------------------------------|
| POST   | `/api/v1/voice-query`  | Full pipeline: audio → transcript → grounded answer   |
| POST   | `/api/v1/query`        | Same pipeline, text input (skips STT; used for benchmarking) |
| POST   | `/api/v1/transcribe`   | STT only, for debugging the mic/STT leg in isolation   |
| GET    | `/api/v1/health`       | Liveness + index/model readiness                       |
| GET    | `/api/v1/metrics`      | P50/P70/P100 latency percentiles per pipeline stage     |

## How each requirement is met

**1. Speech-to-text.** `STT_PROVIDER` selects ElevenLabs Scribe v2 (default)
or Sarvam behind one `BaseTranscriber` interface (`app/utils/stt_client.py`),
so swapping providers is a config change, not a rewrite.

**2. Chunking.** `app/retrieval/chunking.py` implements four strategies —
fixed-size (baseline), sliding-window with overlap, sentence-boundary-aware
semantic grouping (embedding-similarity breakpoints), and a metadata-aware
wrapper that tags every chunk with its source document. `ChunkingPipeline`
runs the configured subset over every document at index-build time; the
resulting redundancy across strategies is intentionally resolved later by
hybrid retrieval + reranking rather than being a modeling weakness.

**3 & 4. Latency target + analytics.** Every request is timed stage-by-stage
with `LatencyTracker` (`app/utils/logger.py`): `guardrail_input`,
`retrieval`, `rerank`, `generation`, `guardrail_grounding` (plus `stt` on the
voice endpoint). Records persist to `data/latency_log.jsonl`; `/api/v1/metrics`
and `scripts/benchmark_latency.py` both compute P50/P70/P100 from real
traffic, not a single best-case run. Honest caveat: the retrieval/chunking/
guardrail stages are local (CPU embeddings + FAISS + BM25) and comfortably fit
inside a 200ms budget, but the **generation** stage makes a real network call
to an external LLM API, which routinely exceeds 200ms on its own — no
external LLM provider can be forced under that ceiling. The benchmark reports
per-stage numbers precisely so this trade-off is visible rather than hidden
in a single blended figure.

**5. Harness.** `app/llm/generator.py` never does a bare prompt-in/text-out
call: the model is forced (via `tool_choice`) to call a `provide_answer` tool
with a typed schema, malformed/missing tool calls and transient errors are
retried with exponential backoff (`tenacity`), and any unrecoverable failure
returns an explicit, safe fallback answer rather than a raw exception or a
guess. `app/api/routes.py` wraps every stage the same way at the orchestration
level.

**6. Guardrails.** Input side (`app/guardrails/safety.py`): regex-based unsafe
content screening, plus embedding-similarity-to-corpus-centroid off-topic
detection, both before any retrieval/generation cost is paid. Output side
(`app/guardrails/grounding.py`): combines the model's own self-reported
`is_answerable`/`confidence` with an independent embedding-overlap check
between the answer and its cited context; if the combined score is below
threshold, the API returns an explicit "I don't have enough grounded
information" response instead of the raw generation — the system is designed
to visibly decline rather than quietly hallucinate.

## Notes / assumptions

- Vector index is local (FAISS + BM25, on-disk under `data/index/`) rather
  than a hosted vector DB, specifically to keep retrieval latency local and
  network-free.
- `DATASET_MAX_DOCS` caps indexing to a demo-scale subset of MSMARCO-XI by
  default (5000 docs); raise it for a fuller corpus at the cost of build time
  and memory.
- `LLM_MODEL` defaults to `claude-sonnet-4-6`; any Anthropic model can be
  substituted via `.env` without code changes.
