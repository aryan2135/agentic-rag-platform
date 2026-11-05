# tests/test_graph.py
import pytest
from app.graph import build_graph
from app.state import initial_state
from tests.fakes import FakeDeps


@pytest.mark.parametrize("scores,mx,rewrites,fallback", [
    ([0.9], 2, 0, False),
    ([0.3, 0.8], 2, 1, False),
    ([0.1, 0.2, 0.3, 0.4], 2, 2, True),
    ([0.1], 0, 0, True),
])
async def test_graph_loop(scores, mx, rewrites, fallback):
    deps = FakeDeps(scores)
    out = await build_graph().ainvoke(
        initial_state("q", mx), config={"configurable": {"deps": deps}, "recursion_limit": 25}
    )
    nodes = [e["node"] for e in out["step_log"]]
    assert out["rewrite_count"] == rewrites and out["fallback_used"] is fallback and deps.last_low is fallback
    assert nodes.count("retrieve") == rewrites + 1 and nodes.count("grade_documents") == rewrites + 1
    assert nodes[-1] == "generate" and len(out["documents"]) == 5
    assert out["original_query"] == "q"

