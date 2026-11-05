# app/state.py
from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict


class RetrievedDoc(TypedDict):
    doc_id: str
    text: str
    source: str
    rrf_score: float                    # fused first-stage score
    rerank_score: NotRequired[float]    # sigmoid-normalised cross-encoder score in [0, 1]


class StepLogEntry(TypedDict):
    node: Literal["retrieve", "grade_documents", "rewrite_query", "generate"]
    ts: float                           # epoch seconds
    duration_ms: float
    detail: dict[str, object]           # JSON-serialisable values only


class GraphState(TypedDict):
    # --- inputs (set once by the caller) ---
    original_query: str
    query: str                          # current query; overwritten by rewrite_query
    max_rewrites: int                   # 0 disables the corrective loop (single-pass baseline)
    # --- retrieval / grading (overwritten each pass; no reducer) ---
    documents: list[RetrievedDoc]
    relevance_score: float | None       # aggregated cross-encoder score in [0, 1]; None before grading
    # --- control ---
    rewrite_count: int
    # --- outputs ---
    answer: NotRequired[str]
    fallback_used: NotRequired[bool]    # True if generate ran with relevance_score < threshold
    # --- observability (append-only via reducer) ---
    step_log: Annotated[list[StepLogEntry], operator.add]


def initial_state(query: str, max_rewrites: int) -> GraphState:
    return {
        "original_query": query,
        "query": query,
        "max_rewrites": max_rewrites,
        "documents": [],
        "relevance_score": None,
        "rewrite_count": 0,
        "step_log": [],
    }

