# scripts/bench_latency.py
import asyncio
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from app.config import Settings
from app.main import create_app


async def benchmark_sequential(client: httpx.AsyncClient, queries: list[str], count: int):
    retrieval_path_times = []
    end_to_end_times = []

    for i in range(count):
        q = queries[i % len(queries)]
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                resp = await client.post("/v1/query", json={"query": q}, timeout=60.0)
                e2e = (time.perf_counter() - t0) * 1000
                end_to_end_times.append(e2e)

                if resp.status_code == 200:
                    data = resp.json()
                    retrieval_ms = sum(
                        entry["duration_ms"]
                        for entry in data.get("step_log", [])
                        if entry.get("node") in {"retrieve", "grade_documents", "rewrite_query"}
                    )
                    retrieval_path_times.append(retrieval_ms)
                else:
                    retrieval_path_times.append(e2e)
                break
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.2)

        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{count} sequential requests...")

    return retrieval_path_times, end_to_end_times


async def benchmark_concurrent(client: httpx.AsyncClient, queries: list[str], count: int, concurrency: int = 8):
    sem = asyncio.Semaphore(concurrency)

    async def fetch(q):
        async with sem:
            for attempt in range(3):
                try:
                    return await client.post("/v1/query", json={"query": q}, timeout=60.0)
                except Exception:
                    if attempt == 2:
                        return None
                    await asyncio.sleep(0.2)

    t0 = time.perf_counter()
    tasks = [fetch(queries[i % len(queries)]) for i in range(count)]
    responses = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t0
    success = sum(1 for r in responses if r is not None and r.status_code == 200)
    throughput = count / total_time if total_time > 0 else 0.0
    return throughput, success


async def main():
    queries_file = "data/eval/questions_clean.jsonl"
    if os.path.exists(queries_file):
        with open(queries_file, encoding="utf-8") as f:
            queries = [json.loads(line)["question"] for line in f if line.strip()]
    else:
        queries = ["What is artificial intelligence?", "How does machine learning work?"]

    cfg = Settings()
    app = create_app()

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            print("Warming up (10 requests)...")
            await benchmark_sequential(client, queries, count=10)

            print("Running 100 sequential requests...")
            retrieval_times, e2e_times = await benchmark_sequential(client, queries, count=100)

            print("Running 100 concurrent requests (concurrency=8)...")
            throughput, success = await benchmark_concurrent(client, queries, count=100, concurrency=8)

    retrieval_p50 = float(np.percentile(retrieval_times, 50))
    retrieval_p95 = float(np.percentile(retrieval_times, 95))
    e2e_p50 = float(np.percentile(e2e_times, 50))
    e2e_p95 = float(np.percentile(e2e_times, 95))

    report = {
        "n_requests": 100,
        "retrieval_path_p50_ms": round(retrieval_p50, 2),
        "retrieval_path_p95_ms": round(retrieval_p95, 2),
        "end_to_end_p50_ms": round(e2e_p50, 2),
        "end_to_end_p95_ms": round(e2e_p95, 2),
        "concurrency_8_throughput_rps": round(throughput, 2),
        "concurrent_success_count": success,
    }

    print("\n=== Latency Benchmark Results ===")
    print(f"Retrieval Path (retrieve+grade+rewrite): p50={retrieval_p50:.1f}ms, p95={retrieval_p95:.1f}ms")
    print(f"End-to-End: p50={e2e_p50:.1f}ms, p95={e2e_p95:.1f}ms")
    print(f"Concurrency-8 Throughput: {throughput:.1f} req/s")

    os.makedirs("reports", exist_ok=True)
    with open("reports/latency_bench.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Saved report to reports/latency_bench.json")


if __name__ == "__main__":
    asyncio.run(main())

