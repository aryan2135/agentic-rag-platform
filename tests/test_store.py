# tests/test_store.py
import numpy as np
from qdrant_client import AsyncQdrantClient
from rank_bm25 import BM25Okapi
from app import qdrant_store as qs


def test_rrf_known():
    s = qs.rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert s["a"] == s["b"] and s["a"] > s["c"] > 0 and s["c"] == s["d"]


async def test_dense_plus_bm25_fusion():
    c = AsyncQdrantClient(":memory:")
    await qs.ensure_collection(c, "t", 8); await qs.ensure_collection(c, "t", 8)  # idempotent
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(5, 8)); texts = ["red apple", "blue sky", "green tree", "red car", "apple pie"]
    ids = [f"d{i}" for i in range(5)]
    await qs.upsert_batch(c, "t", ids, vecs.tolist(), [{"doc_id": i, "text": t} for i, t in zip(ids, texts)])
    pts = await qs.dense_search(c, "t", vecs[0].tolist(), 3)
    assert pts[0].payload["doc_id"] == "d0"
    sc = BM25Okapi([t.split() for t in texts]).get_scores(["apple"])
    bm = [ids[i] for i in np.argsort(-sc)[:3]]
    fused = qs.rrf([[p.payload["doc_id"] for p in pts], bm])
    assert max(fused, key=fused.get) in {"d0", "d4"}
    await c.close()

