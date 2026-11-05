# tests/test_api.py
import httpx
import pytest
from app.config import Settings
from app.main import create_app
from tests.fakes import FakeDeps


async def factory(cfg, client):
    return FakeDeps([0.9])  # tests replace app.state.deps per request


@pytest.fixture
async def client():
    app = create_app(deps_factory=factory, settings=Settings(qdrant_url="", max_rewrites=2))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as cl:
            yield cl, app


async def test_routes(client):
    cl, app = client
    assert (await cl.get("/healthz")).json() == {"ok": True}
    assert (await cl.get("/readyz")).json()["ok"] is True
    app.state.deps = FakeDeps([0.3, 0.9])
    j = (await cl.post("/v1/query", json={"query": "hi"})).json()
    assert j["fallback_used"] is False and j["rewrite_count"] == 1 and len(j["step_log"]) == 6
    assert (await cl.post("/v1/query", json={"query": ""})).status_code == 422
    app.state.deps = FakeDeps([0.3, 0.9])  # fresh stateful fake
    async with cl.stream("POST", "/v1/query/stream", json={"query": "hi"}) as s:
        body = "".join([x async for x in s.aiter_text()])
    assert "event: rewrite_query" in body and body.rstrip().endswith("event: done\ndata: {}")


def test_settings_empty_env(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "")
    assert Settings().qdrant_url is None

