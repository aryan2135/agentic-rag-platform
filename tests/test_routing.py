# tests/test_routing.py
import pytest
from app.routing import decide_to_generate


@pytest.mark.parametrize("score,rc,mx,expected", [
    (0.699999, 0, 2, "rewrite_query"),
    (0.70, 0, 2, "generate"),
    (0.95, 0, 2, "generate"),
    (None, 0, 2, "rewrite_query"),
    (0.0, 0, 2, "rewrite_query"),
    (0.10, 2, 2, "generate"),      # budget exhausted
    (0.10, 1, 2, "rewrite_query"),
    (0.10, 0, 0, "generate"),      # loop disabled (single-pass baseline)
])
def test_decide(score, rc, mx, expected):
    assert decide_to_generate({"relevance_score": score, "rewrite_count": rc, "max_rewrites": mx}) == expected

