# app/routing.py
from __future__ import annotations

from typing import Literal

from app.state import GraphState

RELEVANCE_THRESHOLD: float = 0.70   # spec invariant; do not make configurable


def decide_to_generate(state: GraphState) -> Literal["rewrite_query", "generate"]:
    """Conditional-edge router. Pure: no I/O, no state mutation, no logging."""
    raw = state.get("relevance_score")
    score = 0.0 if raw is None else float(raw)
    if score >= RELEVANCE_THRESHOLD:
        return "generate"
    if state.get("rewrite_count", 0) >= state.get("max_rewrites", 0):
        return "generate"           # retry budget exhausted -> generate() sets fallback_used=True
    return "rewrite_query"

