"""
Fires a batch of text queries at the running API (`/api/v1/query`) and
reports P50 / P70 / P100 latency per pipeline stage -- this is what backs the
latency-analytics numbers required in the submission. It deliberately reuses
real HTTP round trips against a live server rather than calling internal
functions directly, so the numbers reflect what the deployed service actually
does.

Usage:
    # 1. Start the server in one terminal:
    uvicorn app.main:app --port 8000
    # 2. Run the benchmark in another:
    python scripts/benchmark_latency.py --n 50 --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import statistics
import time

import requests

DEFAULT_QUERIES = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "Who wrote the theory of relativity?",
    "What causes inflation in an economy?",
    "How do vaccines train the immune system?",
    "What is the boiling point of water at sea level?",
    "Explain how a search engine ranks web pages.",
    "What is the difference between weather and climate?",
    "How does a car engine convert fuel into motion?",
    "What is the significance of the Magna Carta?",
]


def run(base_url: str, n: int, queries: list[str]) -> None:
    latencies_total = []
    stage_latencies: dict[str, list[float]] = {}
    errors = 0

    for i in range(n):
        query = queries[i % len(queries)]
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{base_url}/api/v1/query", json={"query": query}, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}] request failed: {exc}")
            errors += 1
            continue
        wall_ms = (time.perf_counter() - t0) * 1000

        latencies_total.append(wall_ms)
        for stage, ms in payload.get("latency", {}).get("stages_ms", {}).items():
            stage_latencies.setdefault(stage, []).append(ms)

    def pct(values: list[float]) -> tuple[float, float, float]:
        if not values:
            return (0.0, 0.0, 0.0)
        values_sorted = sorted(values)
        p50 = statistics.median(values_sorted)
        p70_idx = int(0.7 * (len(values_sorted) - 1))
        p70 = values_sorted[p70_idx]
        p100 = values_sorted[-1]
        return p50, p70, p100

    print(f"\n--- Benchmark results ({n - errors}/{n} succeeded) ---")
    print(f"{'stage':<20}{'P50 (ms)':>12}{'P70 (ms)':>12}{'P100 (ms)':>12}")
    for stage, values in sorted(stage_latencies.items()):
        p50, p70, p100 = pct(values)
        print(f"{stage:<20}{p50:>12.1f}{p70:>12.1f}{p100:>12.1f}")

    p50, p70, p100 = pct(latencies_total)
    print(f"{'TOTAL (wall clock)':<20}{p50:>12.1f}{p70:>12.1f}{p100:>12.1f}")
    print(
        "\nNote: TOTAL wall clock includes HTTP + JSON overhead on top of the "
        "server-side 'total_ms' in each response; server-side stage timings "
        "above are the authoritative per-stage numbers."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=50, help="number of test queries to run")
    args = parser.parse_args()
    run(args.base_url, args.n, DEFAULT_QUERIES)
