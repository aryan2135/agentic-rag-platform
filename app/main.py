# app/main.py
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

from app.config import Settings
from app.graph import Deps, build_graph
from app.state import initial_state

DepsFactory = Callable[[Settings, AsyncQdrantClient], Awaitable[Deps]]


class QueryIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class QueryOut(BaseModel):
    answer: str
    relevance_score: float | None
    fallback_used: bool
    rewrite_count: int
    step_log: list[dict[str, object]]


def create_app(deps_factory: DepsFactory | None = None, settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or Settings()
        client = (
            AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key, timeout=10)
            if cfg.qdrant_url
            else AsyncQdrantClient(":memory:")
        )
        factory = deps_factory
        if factory is None:
            from app.deps import build_real_deps  # lazy: heavy imports (torch) stay out of tests
            factory = build_real_deps
        app.state.settings = cfg
        app.state.qdrant = client
        app.state.deps = await factory(cfg, client)
        app.state.graph = build_graph()
        try:
            yield
        finally:
            await client.close()

    app = FastAPI(title="Corrective Agentic RAG", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, object]:
        try:
            cols = await request.app.state.qdrant.get_collections()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"qdrant unavailable: {type(exc).__name__}") from exc
        return {"ok": True, "collections": [c.name for c in cols.collections]}

    def _run_config(request: Request) -> dict[str, object]:
        return {"configurable": {"deps": request.app.state.deps}, "recursion_limit": 25}

    @app.post("/v1/query", response_model=QueryOut)
    async def query(body: QueryIn, request: Request) -> QueryOut:
        st = request.app.state
        try:
            async with asyncio.timeout(60):
                out = await st.graph.ainvoke(initial_state(body.query, st.settings.max_rewrites), config=_run_config(request))
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="pipeline timeout") from exc
        except GraphRecursionError as exc:
            raise HTTPException(status_code=500, detail="graph recursion limit hit") from exc
        return QueryOut(
            answer=out["answer"], relevance_score=out["relevance_score"], fallback_used=out["fallback_used"],
            rewrite_count=out["rewrite_count"], step_log=out["step_log"],
        )

    @app.post("/v1/query/stream")
    async def query_stream(body: QueryIn, request: Request) -> StreamingResponse:
        st = request.app.state

        async def events() -> AsyncIterator[str]:
            async for chunk in st.graph.astream(
                initial_state(body.query, st.settings.max_rewrites), config=_run_config(request), stream_mode="updates"
            ):
                for node, upd in chunk.items():
                    slim = {k: v for k, v in upd.items() if k != "documents"}
                    yield f"event: {node}\ndata: {json.dumps(slim, default=str)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app

