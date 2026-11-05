# tests/fakes.py
from app.state import RetrievedDoc


class FakeDeps:
    """Stateful: each instance serves ONE request. Create a fresh instance per request."""
    def __init__(self, scores: list[float]):
        self.scores, self.calls, self.last_low = scores, 0, None

    async def retrieve(self, query: str) -> list[RetrievedDoc]:
        return [{"doc_id": str(i), "text": f"t{i}", "source": "s", "rrf_score": 0.1} for i in range(20)]

    async def rerank(self, query, docs):
        s = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return [{**d, "rerank_score": s} for d in docs]

    async def rewrite(self, original_query, query, docs): return query + " v2"

    async def generate(self, query, docs, low_confidence):
        self.last_low = low_confidence
        return "ans"

