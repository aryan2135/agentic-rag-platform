# app/graph.py
from __future__ import annotations

import time
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.routing import RELEVANCE_THRESHOLD, decide_to_generate
from app.state import GraphState, RetrievedDoc, StepLogEntry

TOP_K_RERANK = 5


class Deps(Protocol):
    """Everything the nodes need. Real impl: app/deps.py. Test impl: fakes in tests/."""
    async def retrieve(self, query: str) -> list[RetrievedDoc]: ...
    async def rerank(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]: ...   # sorted desc by rerank_score
    async def rewrite(self, original_query: str, query: str, docs: list[RetrievedDoc]) -> str: ...
    async def generate(self, query: str, docs: list[RetrievedDoc], low_confidence: bool) -> str: ...


def _deps(config: RunnableConfig) -> Deps:
    return config["configurable"]["deps"]


def _log(node: str, t0: float, **detail: object) -> list[StepLogEntry]:
    return [{"node": node, "ts": time.time(), "duration_ms": (time.perf_counter() - t0) * 1000, "detail": detail}]  # type: ignore[typeddict-item]


async def retrieve(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    t0 = time.perf_counter()
    docs = await _deps(config).retrieve(state["query"])
    return {"documents": docs, "step_log": _log("retrieve", t0, n=len(docs), query=state["query"])}


async def grade_documents(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    t0 = time.perf_counter()
    ranked = await _deps(config).rerank(state["query"], state["documents"])
    kept = ranked[:TOP_K_RERANK]
    score = sum(d["rerank_score"] for d in kept) / len(kept) if kept else 0.0
    return {
        "documents": kept,
        "relevance_score": score,
        "step_log": _log("grade_documents", t0, kept=len(kept), relevance_score=round(score, 4)),
    }


async def rewrite_query(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    t0 = time.perf_counter()
    new_q = await _deps(config).rewrite(state["original_query"], state["query"], state["documents"])
    return {
        "query": new_q,
        "rewrite_count": state["rewrite_count"] + 1,
        "step_log": _log("rewrite_query", t0, old=state["query"], new=new_q),
    }


async def generate(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    t0 = time.perf_counter()
    low = (state.get("relevance_score") or 0.0) < RELEVANCE_THRESHOLD
    answer = await _deps(config).generate(state["original_query"], state["documents"], low)
    return {"answer": answer, "fallback_used": low, "step_log": _log("generate", t0, fallback_used=low)}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", END)
    return g.compile()

