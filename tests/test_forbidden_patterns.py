# tests/test_forbidden_patterns.py
import pathlib
import re

RULES = {
    r"(?<!Async)QdrantClient\(": "sync QdrantClient is forbidden; use AsyncQdrantClient",
    r"\.search\(": "client.search is deprecated; use await client.query_points",
    r"on_event\(": "use lifespan, not @app.on_event",
    r"time\.sleep\(": "blocking sleep in async code",
    r"import requests|from requests": "use httpx.AsyncClient",
    r"(?<!a)invoke\(": "use graph.ainvoke / llm.ainvoke in async code",
    r"asyncio\.run\(|run_until_complete\(": "never start a loop inside the app",
    r"\bTODO\b|NotImplementedError": "no placeholders",
}


def test_no_forbidden_patterns():
    hits = []
    for p in pathlib.Path("app").rglob("*.py"):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for pat, why in RULES.items():
                if re.search(pat, line):
                    hits.append(f"{p}:{n}: {why}: {line.strip()}")
    assert not hits, "\n".join(hits)

