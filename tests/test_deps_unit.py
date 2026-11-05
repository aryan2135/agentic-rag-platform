# tests/test_deps_unit.py
import json
import pytest
from qdrant_client import AsyncQdrantClient

from app.config import Settings
from app.deps import Bm25Index, RealDeps
from app.state import RetrievedDoc
from app import qdrant_store as qs


class FakeEmbedder:
    dim: int = 8

    async def aencode_query(self, query: str) -> list[float]:
        return [0.1] * self.dim

    async def aencode_docs(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]


class FakeScorer:
    async def ascore(self, query: str, texts: list[str]) -> list[float]:
        # Return deterministic descending or arbitrary scores
        return [float(i) / max(len(texts), 1) for i in range(len(texts))]


class FakeBm25:
    def __init__(self, items: list[tuple[str, float]]) -> None:
        self.items = items
        self.payloads = {doc_id: {"doc_id": doc_id, "text": f"text_{doc_id}", "source": "test"} for doc_id, _ in items}

    def payload(self, doc_id: str) -> dict:
        return self.payloads[doc_id]

    async def atop_k(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.items[:k]


async def test_bm25_index_unit(tmp_path):
    corpus_path = tmp_path / "bm25_test.jsonl"
    sample_docs = [
        {"doc_id": "1", "text": "The quick brown fox jumps over the lazy dog", "source": "s"},
        {"doc_id": "2", "text": "Artificial intelligence and machine learning are revolutionizing technology", "source": "s"},
        {"doc_id": "3", "text": "Apples, bananas, and oranges are popular fruits", "source": "s"},
        {"doc_id": "4", "text": "Deep learning models require large training datasets", "source": "s"},
        {"doc_id": "5", "text": "Quantum computers use qubits instead of classical bits", "source": "s"},
    ]
    with open(corpus_path, "w", encoding="utf-8") as f:
        for d in sample_docs:
            f.write(json.dumps(d) + "\n")

    idx = Bm25Index(str(corpus_path))
    res = await idx.atop_k("machine learning", 3)
    assert len(res) > 0
    assert res[0][0] == "2"

    empty_res = await idx.atop_k("", 3)
    assert empty_res == []

    payload = idx.payload("2")
    assert payload["doc_id"] == "2"
    assert "machine learning" in payload["text"]


async def test_real_deps_retrieve_unit():
    client = AsyncQdrantClient(":memory:")
    cfg = Settings(qdrant_collection="test_col", top_k_dense=10, top_k_bm25=10, top_k_fused=5, rrf_k=60)
    await qs.ensure_collection(client, cfg.qdrant_collection, 8)

    # Upsert some test points in Qdrant
    doc_ids = [f"d_{i}" for i in range(10)]
    vectors = [[0.1] * 8 for _ in range(10)]
    payloads = [{"doc_id": d, "text": f"content_{d}", "source": "src"} for d in doc_ids]
    await qs.upsert_batch(client, cfg.qdrant_collection, doc_ids, vectors, payloads)

    bm25_hits = [(f"d_{i}", 1.0) for i in range(5, 15)]
    fake_bm25 = FakeBm25(bm25_hits)
    fake_embedder = FakeEmbedder()
    fake_scorer = FakeScorer()

    deps = RealDeps(
        cfg=cfg,
        client=client,
        embedder=fake_embedder,  # type: ignore[arg-type]
        bm25=fake_bm25,          # type: ignore[arg-type]
        scorer=fake_scorer,      # type: ignore[arg-type]
        llm=None,                # type: ignore[arg-type]
    )

    retrieved = await deps.retrieve("query")
    assert len(retrieved) <= cfg.top_k_fused
    # Check that retrieved items are sorted by rrf_score descending
    scores = [d["rrf_score"] for d in retrieved]
    assert scores == sorted(scores, reverse=True)
    await client.close()


async def test_real_deps_rerank_unit():
    docs: list[RetrievedDoc] = [
        {"doc_id": "1", "text": "one", "source": "s", "rrf_score": 0.5},
        {"doc_id": "2", "text": "two", "source": "s", "rrf_score": 0.4},
        {"doc_id": "3", "text": "three", "source": "s", "rrf_score": 0.3},
    ]

    class MockScorer:
        async def ascore(self, query: str, texts: list[str]) -> list[float]:
            # Give higher score to 'two', then 'three', then 'one'
            mapping = {"one": 0.2, "two": 0.9, "three": 0.7}
            return [mapping[t] for t in texts]

    deps = RealDeps(
        cfg=Settings(),
        client=None,            # type: ignore[arg-type]
        embedder=None,          # type: ignore[arg-type]
        bm25=None,              # type: ignore[arg-type]
        scorer=MockScorer(),    # type: ignore[arg-type]
        llm=None,               # type: ignore[arg-type]
    )

    ranked = await deps.rerank("test query", docs)
    assert len(ranked) == 3
    assert ranked[0]["doc_id"] == "2"
    assert ranked[0]["rerank_score"] == 0.9
    assert ranked[1]["doc_id"] == "3"
    assert ranked[1]["rerank_score"] == 0.7
    assert ranked[2]["doc_id"] == "1"
    assert ranked[2]["rerank_score"] == 0.2

