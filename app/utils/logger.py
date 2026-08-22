"""
Structured logging + latency instrumentation.

Every pipeline stage (stt, chunking, retrieval, rerank, guardrails, generation)
is timed via `LatencyTracker`. Per-request breakdowns are appended to a JSONL
file and can be aggregated into P50 / P70 / P100 percentiles per stage via
`compute_percentiles`, which backs the /api/v1/metrics endpoint and the
latency numbers required in the submission.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np

from app.config import get_settings

_settings = get_settings()


def setup_logger(name: str = "swan") -> logging.Logger:
    """Configure a structured (JSON-ish) logger that writes to stdout.

    Using stdout keeps this container-friendly (no log file management needed
    for the app logs themselves; latency numbers are persisted separately).
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on reload
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


@dataclass
class LatencyTracker:
    """Times pipeline stages for a single request and persists the result.

    Usage:
        tracker = LatencyTracker(request_id="abc123")
        with tracker.stage("retrieval"):
            ...
        tracker.finalize()
    """

    request_id: str
    stages: dict[str, float] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stages[name] = round(elapsed_ms, 3)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 3)

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "stages_ms": self.stages,
            "total_ms": self.total_ms,
        }

    def finalize(self) -> dict:
        """Log + persist this request's latency breakdown, return it."""
        record = self.as_dict()
        record["ts"] = time.time()
        logger.info(json.dumps({"latency": record}))
        _persist(record)
        return record


_write_lock = Lock()


def _persist(record: dict) -> None:
    path = Path(_settings.LATENCY_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def compute_percentiles(limit: int = 500) -> dict:
    """Aggregate the last `limit` requests into P50/P70/P100 per stage + total.

    P100 is the max observed value (worst case), which is what the task asks
    for alongside P50/P70 rather than a true "100th percentile" statistical
    estimate.
    """
    path = Path(_settings.LATENCY_LOG_PATH)
    if not path.exists():
        return {"count": 0, "stages": {}, "total": None}

    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records = records[-limit:]

    if not records:
        return {"count": 0, "stages": {}, "total": None}

    def pct(values: list[float]) -> dict:
        arr = np.array(values, dtype=float)
        return {
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p70": round(float(np.percentile(arr, 70)), 3),
            "p100": round(float(np.max(arr)), 3),
            "mean": round(float(np.mean(arr)), 3),
        }

    stage_names = set()
    for r in records:
        stage_names.update(r.get("stages_ms", {}).keys())

    stages_out = {}
    for name in sorted(stage_names):
        values = [r["stages_ms"][name] for r in records if name in r.get("stages_ms", {})]
        if values:
            stages_out[name] = pct(values)

    totals = [r["total_ms"] for r in records if "total_ms" in r]

    return {
        "count": len(records),
        "stages": stages_out,
        "total": pct(totals) if totals else None,
    }
