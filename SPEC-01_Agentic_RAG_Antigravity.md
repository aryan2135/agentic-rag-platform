# SPEC-01 — Autonomous Corrective Agentic RAG Platform

Role: Staff AI Architect specification. Consumer: autonomous coding agent. Execute top to bottom. Do not ask clarifying questions; where a decision is unspecified, apply the stated default and record it in `DECISIONS.md`.

Stack: Python 3.12, LangGraph, FastAPI, Qdrant (async client), Gemini via LangChain, local embedding + cross-encoder models.

---

## 0. Operating rules (apply to every step)

| # | Rule |
|---|------|
| R1 | Read this entire file before writing any code. |
| R2 | Phase gates (§13). Do not start phase N+1 until every gate command of phase N exits 0. Append each gate command and its output to `reports/gate_log.md`. |
| R3 | No fabricated APIs. After installing dependencies, introspect anything uncertain (`python -c "import X; print(X.__version__)"`, `inspect.signature(...)`, `help(...)`). Installed version wins over memory or documentation. |
| R4 | No placeholders in `app/`: no `TODO`, `NotImplementedError`, `pass`-only bodies, hard-coded fake data. (`tests/test_forbidden_patterns.py` enforces this.) |
| R5 | Offline-first tests. The default `pytest` run must pass with no network, no API keys, no model downloads (fakes + `AsyncQdrantClient(":memory:")`). Tests needing network carry `@pytest.mark.network` and are skipped by default. |
| R6 | Never block the event loop. Sync/CPU-bound work (BM25 scoring, embedding, cross-encoder inference, file I/O in request path) runs via `await asyncio.to_thread(...)`. |
| R7 | Never edit measured numbers to match targets. Targets in §1 are hypotheses. Reports contain measured values only. |
| R8 | On a failing command: read the traceback, fix root cause, rerun. Max 3 attempts per distinct failure, then log it in `reports/blockers.md` and continue with independent work. |
| R9 | Secrets only via environment. `.env` is git-ignored; `.env.example` is committed. |
| R10 | Do not change the public interfaces defined in §4, §5, §7, §8. Extend only. |

---

## 1. Target definition and claim-to-metric mapping

The system must implement the resume claims below. Each claim is bound to one measurable definition. Implement the measurement; report the measured value.

| Resume claim | Implementation | Metric definition (use exactly this) |
|---|---|---|
| Two-stage hybrid retriever (BM25 + Qdrant via RRF) + Cross-Encoder, pruning 20 → 5 | §6 | `top_k_fused=20` → cross-encoder → `top_k_rerank=5` |
| Boosting recall by 34% | §6 | Relative gain `(R_full − R_dense) / R_dense`, where `R` = Recall@5 (gold doc ∈ top 5). `R_dense` = Qdrant dense-only top 5. `R_full` = hybrid + RRF + cross-encoder top 5, single pass. |
| Corrective loop: rewrite when relevance < 70% | §5 | `relevance_score` = mean sigmoid-normalised cross-encoder score of the 5 kept docs; `< 0.70` → `rewrite_query`. |
| Reducing retrieval errors by 62% | §5, §12 | Retrieval error = gold doc ∉ final top 5. Reduction = `1 − err_loop / err_single`, where `err_single` is the same pipeline with `max_rewrites=0`. Report on the `clean` and `degraded` slices separately. |
| Ragas, 200+ questions, 91% accuracy | §12 | Report each Ragas metric by its real name (faithfulness, answer_relevancy, answer_correctness …). Also report deterministic `gold_answer_contained` (normalised gold span ⊂ answer). Never call a metric "accuracy" unless it is. |
| FastAPI at sub-800 ms latency | §7, §12 | p95 over ≥100 warm sequential requests. Report two numbers: `retrieval_path_p95_ms` (retrieve + grade + rewrite nodes) and `end_to_end_p95_ms` (includes LLM generation). |

If measured values differ from the resume numbers, the final report states the measured values and lists the resume bullets that need editing. Do not tune the system toward the resume numbers.

---

## 2. Environment

- Python **3.12** (3.11 minimum: LangGraph needs it for async `RunnableConfig` propagation).
- `python -m venv .venv`; install; then `pip freeze > requirements.lock.txt` and record versions in `DECISIONS.md`.
- Versions this design was verified against: `langgraph 1.2.x`, `qdrant-client 1.20`, `fastapi 0.16x`, `pydantic 2.x`, `pydantic-settings 2.x`, `pytest-asyncio 1.x`, `rank-bm25 0.2.2`.

`requirements.txt`:
```
langgraph
langchain-core
langchain-google-genai
fastapi
uvicorn[standard]
qdrant-client
pydantic
pydantic-settings
rank-bm25
sentence-transformers
transformers
torch
numpy
datasets
ragas
httpx
pytest
pytest-asyncio
```

`.env.example`:
```
GOOGLE_API_KEY=
LLM_MODEL=gemini-2.5-flash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```
`LLM_MODEL` is read from the environment; verify the name against the provider's current model list before the first live call.

`docker-compose.yml` (dev only; tests use `:memory:`):
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest   # after first pull, pin the tag you pulled; record it in DECISIONS.md
    ports: ["6333:6333"]
    volumes: ["./qdrant_storage:/qdrant/storage"]
```

---

## 3. Repository layout

```
agentic-rag/
├── app/
│   ├── __init__.py
│   ├── config.py          # Settings (pydantic-settings)
│   ├── state.py           # GraphState schema            (§4)
│   ├── routing.py         # decide_to_generate           (§5)
│   ├── graph.py           # nodes + build_graph, Deps protocol (§5)
│   ├── qdrant_store.py    # async Qdrant helpers + RRF   (§6, §7)
│   ├── deps.py            # RealDeps, Embedder, Bm25Index, CrossEncoderScorer, LLM wrappers (§6)
│   └── main.py            # create_app() FastAPI factory (§7)
├── scripts/
│   ├── ingest.py          # corpus → Qdrant + bm25_corpus.jsonl
│   ├── build_eval_data.py # SQuAD-based eval corpus + clean/degraded questions
│   ├── eval_retrieval.py  # recall / retrieval-error experiments
│   ├── eval_ragas.py      # Ragas + gold_answer_contained
│   └── bench_latency.py   # p50/p95
├── tests/                 # conftest.py, fakes.py, test_*.py (§11)
├── data/  reports/
├── pytest.ini  requirements.txt  requirements.lock.txt
├── .env.example  docker-compose.yml  Makefile  README.md  DECISIONS.md
```

Use the flat `app/` package exactly as named (imports are `from app.x import y`; `pytest.ini` sets `pythonpath = .`).

---

## 4. `GraphState` schema (production-grade)

Create `app/state.py` verbatim:

```python
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
```

Schema rules:
1. `GraphState` is a `TypedDict` (native LangGraph state; reducers work on it). Do not convert to Pydantic.
2. `step_log` is the only reducer field (`operator.add`). Nodes return a **list** with one entry; LangGraph appends. Never mutate state in place; nodes return partial-update dicts.
3. `documents` and `relevance_score` carry no reducer: each retrieval/grading pass overwrites them.
4. `original_query` is immutable after `initial_state()`. `query` changes only in `rewrite_query`. `generate` answers `original_query`.
5. `relevance_score` is in `[0, 1]` after sigmoid normalisation (§6). `None` only before the first grading.
6. Every `step_log[*].detail` value must be JSON-serialisable.

---

## 5. Graph topology, nodes, routing

```
START → retrieve → grade_documents ──decide_to_generate──┬─ "generate" ──────────→ END
            ▲                                            └─ "rewrite_query" ─┐
            └──────────────────────────────────────────────────────────────────┘
```

Node contracts:

| Node | Reads | Writes | Behaviour |
|---|---|---|---|
| `retrieve` | `query` | `documents`, `step_log` | Hybrid retrieval → fused top 20. |
| `grade_documents` | `query`, `documents` | `documents` (top 5, sorted desc), `relevance_score`, `step_log` | Cross-encoder rescoring; `relevance_score` = mean of kept scores; empty → `0.0`. |
| `rewrite_query` | `original_query`, `query`, `documents` | `query`, `rewrite_count` (+1), `step_log` | LLM rewrite; empty or identical output keeps the current query (the counter still increments, so the loop is bounded). |
| `generate` | `original_query`, `documents`, `relevance_score` | `answer`, `fallback_used`, `step_log` | `fallback_used = relevance_score < 0.70`; prompt tells the LLM to state uncertainty when true. |

### 5.1 Routing function — `decide_to_generate`

Create `app/routing.py` verbatim:

```python
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
```

Routing contract:
- `score < 0.70` and retry budget remaining → `"rewrite_query"`.
- `score >= 0.70` → `"generate"`. Boundary `0.70` itself passes.
- `relevance_score is None` is treated as `0.0`.
- `rewrite_count >= max_rewrites` → `"generate"` (guarantees termination). `max_rewrites=0` disables the loop (single-pass baseline for §12).
- The function is pure: no I/O, no logging, no mutation. `RELEVANCE_THRESHOLD` is a module constant, not configuration.
- API default `max_rewrites=2`. Set `recursion_limit=25` on every invocation as a backstop.

### 5.2 Nodes and graph assembly

Create `app/graph.py` verbatim:

```python
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
```

Rules:
- Nodes are `async def (state, config: RunnableConfig)`. Dependencies arrive only via `config["configurable"]["deps"]`. No module-level singletons, no global clients.
- The conditional-edge mapping dict must list both destinations explicitly.

---

## 6. Retrieval, fusion, reranking

### 6.1 Procedure inside `RealDeps.retrieve`
1. Embed the query (`BAAI/bge-small-en-v1.5`, normalised, BGE query prefix `"Represent this sentence for searching relevant passages: "`) via `to_thread`.
2. `asyncio.gather` of: Qdrant dense search (top 50) and BM25 top 50 (`to_thread`).
3. RRF fusion (`k=60`) of the two best-first id lists → top 20 → `RetrievedDoc` list (`rrf_score` set).

```python
async def retrieve(self, query: str) -> list[RetrievedDoc]:
    qvec = await self.embedder.aencode_query(query)
    dense_pts, bm25_hits = await asyncio.gather(
        qs.dense_search(self.client, self.cfg.qdrant_collection, qvec, self.cfg.top_k_dense),
        self.bm25.atop_k(query, self.cfg.top_k_bm25),
    )
    fused = qs.rrf([[p.payload["doc_id"] for p in dense_pts], [d for d, _ in bm25_hits]], k=self.cfg.rrf_k)
    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: self.cfg.top_k_fused]
    payloads = {p.payload["doc_id"]: p.payload for p in dense_pts}
    for d, _ in bm25_hits:
        payloads.setdefault(d, self.bm25.payload(d))
    return [{"doc_id": d, "text": payloads[d]["text"], "source": payloads[d].get("source", ""), "rrf_score": s} for d, s in top]
```

### 6.2 Cross-encoder scoring — mandatory normalisation
Cross-encoder outputs are **unbounded logits, not probabilities**. The 70% threshold is only meaningful after sigmoid. Do **not** rely on `sentence_transformers.CrossEncoder` activation defaults (the kwarg names and defaults changed across versions). Use `transformers` directly:

```python
import asyncio
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderScorer:
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        if self._model.config.num_labels != 1:
            raise ValueError("expected a single-logit relevance model")
        self._max_length = max_length

    @torch.inference_mode()
    def _score_sync(self, query: str, texts: list[str]) -> list[float]:
        enc = self._tok([query] * len(texts), texts, padding=True, truncation="only_second",
                        max_length=self._max_length, return_tensors="pt")
        logits = self._model(**enc).logits.squeeze(-1)   # raw, unbounded
        return torch.sigmoid(logits).tolist()            # -> [0, 1]

    async def ascore(self, query: str, texts: list[str]) -> list[float]:
        return await asyncio.to_thread(self._score_sync, query, texts)
```
Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Network test (`@pytest.mark.network`): a clearly relevant pair scores > 0.5, an unrelated pair scores < 0.5, all scores in `[0, 1]`.

### 6.3 BM25 and embedder wrappers
```python
_TOKEN = re.compile(r"\w+")
def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())

class Bm25Index:
    def __init__(self, path: str) -> None:               # build via await asyncio.to_thread(Bm25Index, path)
        with open(path, encoding="utf-8") as f:
            self._docs = [json.loads(line) for line in f if line.strip()]
        self._ids = [d["doc_id"] for d in self._docs]
        self._by_id = {d["doc_id"]: d for d in self._docs}
        self._bm25 = BM25Okapi([_tok(d["text"]) for d in self._docs])

    def payload(self, doc_id: str) -> dict:
        return self._by_id[doc_id]

    def _top_k(self, query: str, k: int) -> list[tuple[str, float]]:
        toks = _tok(query)
        if not toks:                                      # empty/stopword-only query guard
            return []
        scores = self._bm25.get_scores(toks)
        idx = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i])) for i in idx if scores[i] > 0]

    async def atop_k(self, query: str, k: int) -> list[tuple[str, float]]:
        return await asyncio.to_thread(self._top_k, query, k)
```
`Embedder`: wraps `SentenceTransformer`; `dim = model.get_sentence_embedding_dimension()`; `encode(..., normalize_embeddings=True, show_progress_bar=False)`; async wrappers `aencode_docs(list[str])` and `aencode_query(str)` via `to_thread`. Collection vector size must equal `Embedder.dim` (asserted in `ensure_collection` call site).

### 6.4 LLM wrapper
- `ChatGoogleGenerativeAI(model=cfg.llm_model, temperature=0, google_api_key=cfg.google_api_key)`; call only with `await llm.ainvoke(...)` inside `async with asyncio.timeout(cfg.llm_timeout_s)`.
- `AIMessage.content` may be `str` or a list of content blocks. Normalise: if list, join the `text` of blocks whose `type == "text"`.
- Fail fast at startup (clear error) if `google_api_key` is missing.

Prompts (store as module constants):

`REWRITE_PROMPT`:
```
You rewrite search queries for a retrieval system.
Original question: {original}
Current query: {current}
Top retrieved passages (low relevance):
{snippets}
Write ONE improved search query: more specific, keyword-rich, same intent. Output the query only. No quotes, no explanation.
```
Post-process: first line, strip quotes/whitespace, truncate to 300 chars; empty → return `current`.

`GENERATE_PROMPT`:
```
Answer the question using ONLY the context. Cite sources as [doc_id]. If the context does not contain the answer, say you cannot find it in the provided documents.
{low_confidence_note}
Question: {question}
Context:
{context}
```
`low_confidence_note` when `low_confidence=True`: `Retrieval confidence is low; state your uncertainty explicitly.` else empty. `context` = `[doc_id] text` per kept doc.

### 6.5 Lifespan factory for real dependencies (`app/deps.py::build_real_deps`)
`async def build_real_deps(cfg: Settings, client: AsyncQdrantClient) -> RealDeps`:
1. Build `Embedder`, `CrossEncoderScorer`, `Bm25Index` via `await asyncio.to_thread(...)` (model loading is blocking).
2. `await qs.ensure_collection(client, cfg.qdrant_collection, embedder.dim)`.
3. Warm up: one dummy `aencode_query` and one dummy `ascore` (removes first-request latency).
4. Return `RealDeps(cfg, client, embedder, bm25, scorer, llm)` implementing the `Deps` protocol of §5.2.

---

## 7. Async requirements (strict)

### 7.1 MUST
1. Qdrant: **`AsyncQdrantClient` only**. One client per process, created in the FastAPI lifespan, stored on `app.state`, closed with `await client.close()` in the lifespan `finally`.
2. Qdrant calls are awaited: `await client.collection_exists(...)`, `await client.create_collection(...)`, `await client.upsert(..., wait=True)`, `await client.query_points(...)` (read `.points` from the result), `await client.get_collections()`.
3. FastAPI: lifespan context manager (`@asynccontextmanager`), `async def` routes, Pydantic v2 models for request and response.
4. LangGraph: `await graph.ainvoke(...)` and `graph.astream(..., stream_mode="updates")` consumed with `async for`. Nodes are `async def`.
5. Blocking work (§0 R6) through `asyncio.to_thread`. Independent I/O through `asyncio.gather`.
6. Every external await is bounded: `asyncio.timeout(...)` (LLM: `cfg.llm_timeout_s`; pipeline: 60 s). Map `TimeoutError` → HTTP 504, Qdrant failure → 503, `GraphRecursionError` → 500.
7. Qdrant point ids are UUIDs derived from `doc_id` (`uuid5`); arbitrary string ids are rejected by Qdrant.
8. Tests run the app with `httpx.AsyncClient(transport=httpx.ASGITransport(app=app))` inside `app.router.lifespan_context(app)`.

### 7.2 MUST NOT (enforced by `tests/test_forbidden_patterns.py`)
- `QdrantClient(` (sync), `client.search(` (deprecated; use `query_points`)
- `@app.on_event(` (use lifespan)
- `requests` (use `httpx.AsyncClient`), `time.sleep(`
- `graph.invoke(` / `llm.invoke(` in async code
- `asyncio.run(` / `run_until_complete(` anywhere under `app/`
- module-level global clients or models

### 7.3 Qdrant helpers — create `app/qdrant_store.py` verbatim

```python
# app/qdrant_store.py
from __future__ import annotations

import uuid
from typing import Sequence

from qdrant_client import AsyncQdrantClient, models

_NS = uuid.UUID("6f1c3b9e-5a57-4f0e-9d0b-2f7f7f7f0001")


def point_id(doc_id: str) -> str:
    """Qdrant ids must be uint or UUID. Derive a stable UUID from the external doc_id."""
    return str(uuid.uuid5(_NS, doc_id))


async def ensure_collection(client: AsyncQdrantClient, name: str, dim: int) -> None:
    if not await client.collection_exists(name):
        await client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )


async def upsert_batch(
    client: AsyncQdrantClient, name: str, doc_ids: Sequence[str], vectors: Sequence[Sequence[float]], payloads: Sequence[dict]
) -> None:
    points = [
        models.PointStruct(id=point_id(d), vector=list(v), payload=p)
        for d, v, p in zip(doc_ids, vectors, payloads, strict=True)
    ]
    await client.upsert(collection_name=name, points=points, wait=True)


async def dense_search(client: AsyncQdrantClient, name: str, qvec: Sequence[float], limit: int) -> list[models.ScoredPoint]:
    res = await client.query_points(collection_name=name, query=list(qvec), limit=limit, with_payload=True)
    return res.points


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion. Each ranking is a best-first list of doc_ids."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
```

### 7.4 Settings — create `app/config.py` verbatim

```python
# app/config.py
from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str | None = None          # None -> AsyncQdrantClient(":memory:")
    qdrant_api_key: str | None = None
    qdrant_collection: str = "docs"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gemini-2.5-flash"
    google_api_key: str | None = None
    bm25_corpus_path: str = "data/bm25_corpus.jsonl"
    top_k_dense: int = 50
    top_k_bm25: int = 50
    top_k_fused: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60
    max_rewrites: int = 2
    llm_timeout_s: float = 20.0

    @field_validator("qdrant_url", "qdrant_api_key", "google_api_key", mode="before")
    @classmethod
    def _empty_to_none(cls, v: object) -> object:
        return None if isinstance(v, str) and v.strip() == "" else v
```

### 7.5 FastAPI application — create `app/main.py` verbatim

```python
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
```

Create `app/asgi.py` (separate module so importing `app.main` in tests has no side effects):
```python
from app.main import create_app

app = create_app()
```
Run with `uvicorn app.asgi:app --port 8000`.

---

## 8. API contract

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/healthz` | – | `{"ok": true}` |
| GET | `/readyz` | – | `{"ok": true, "collections": [...]}`; 503 if Qdrant unreachable |
| POST | `/v1/query` | `{"query": str (1–2000 chars)}` | `QueryOut`: `answer`, `relevance_score`, `fallback_used`, `rewrite_count`, `step_log` |
| POST | `/v1/query/stream` | same | SSE; one event per node update (`event: <node>`), final `event: done`; `documents` omitted from payload |

Empty query → 422.

---

## 9. Ingestion (`scripts/ingest.py`)

Input: `data/corpus.jsonl`, one object per line: `{"doc_id": str, "text": str, "source": str}`.
Behaviour (all async, one `AsyncQdrantClient`):
1. `ensure_collection` with `Embedder.dim`.
2. Embed in batches of 64 (`aencode_docs`), `upsert_batch` per batch (idempotent: ids derive from `doc_id`).
3. Write `data/bm25_corpus.jsonl` (same records) — the BM25 index source.
4. Print counts; assert `client.count(collection, exact=True).count == len(corpus)`.
Entry point: `asyncio.run(main())` is allowed in `scripts/` only.

---

## 10. Config reference

| Env var | Default | Notes |
|---|---|---|
| `QDRANT_URL` | empty → in-memory | empty string is coerced to `None` |
| `QDRANT_COLLECTION` | `docs` | |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | |
| `LLM_MODEL` | `gemini-2.5-flash` | verify availability |
| `TOP_K_DENSE` / `TOP_K_BM25` | 50 / 50 | |
| `TOP_K_FUSED` | 20 | |
| `RRF_K` | 60 | |
| `MAX_REWRITES` | 2 | |
| `LLM_TIMEOUT_S` | 20 | |

`TOP_K_RERANK = 5` lives in `app/graph.py` (spec invariant "20 → 5").

---

## 11. Tests (offline by default)

`pytest.ini`:
```
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
pythonpath = .
testpaths = tests
markers =
    network: needs internet / model downloads / API keys (skipped by default)
```

`tests/conftest.py`:
```python
# tests/conftest.py
import pytest


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="network test; run with -m network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
```

`tests/fakes.py` (stateful fake: **one instance per request**; reusing an instance across requests silently changes scores):
```python
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
```

`tests/test_routing.py`:
```python
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
```

`tests/test_graph.py`:
```python
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
```

`tests/test_store.py`:
```python
# tests/test_store.py
import numpy as np
from qdrant_client import AsyncQdrantClient
from rank_bm25 import BM25Okapi
from app import qdrant_store as qs


def test_rrf_known():
    s = qs.rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert s["a"] == s["b"] and s["a"] > s["c"] > 0 and s["c"] == s["d"]


async def test_dense_plus_bm25_fusion():
    c = AsyncQdrantClient(":memory:")
    await qs.ensure_collection(c, "t", 8); await qs.ensure_collection(c, "t", 8)  # idempotent
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(5, 8)); texts = ["red apple", "blue sky", "green tree", "red car", "apple pie"]
    ids = [f"d{i}" for i in range(5)]
    await qs.upsert_batch(c, "t", ids, vecs.tolist(), [{"doc_id": i, "text": t} for i, t in zip(ids, texts)])
    pts = await qs.dense_search(c, "t", vecs[0].tolist(), 3)
    assert pts[0].payload["doc_id"] == "d0"
    sc = BM25Okapi([t.split() for t in texts]).get_scores(["apple"])
    bm = [ids[i] for i in np.argsort(-sc)[:3]]
    fused = qs.rrf([[p.payload["doc_id"] for p in pts], bm])
    assert max(fused, key=fused.get) in {"d0", "d4"}
    await c.close()
```

`tests/test_api.py`:
```python
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
```

`tests/test_forbidden_patterns.py`:
```python
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
        for n, line in enumerate(p.read_text().splitlines(), 1):
            for pat, why in RULES.items():
                if re.search(pat, line):
                    hits.append(f"{p}:{n}: {why}: {line.strip()}")
    assert not hits, "\n".join(hits)
```

Additional tests to write:
- `test_deps_unit.py`: `Bm25Index` on a 5-doc temp file (relevant doc first, empty-query → `[]`); `RealDeps.retrieve` with a fake embedder + `:memory:` Qdrant returns ≤ 20 docs sorted by `rrf_score`; `RealDeps.rerank` with a fake scorer returns docs sorted desc.
- `test_cross_encoder_network.py` (`@pytest.mark.network`): §6.2 checks.
- `test_llm_text_normalisation.py`: str content and block-list content both yield `str`.

Expected baseline: the files above yield **17 passing tests** before the additional ones; the suite must stay green after every phase.

---

## 12. Evaluation harness

All eval scripts write machine-readable JSON to `reports/` and a human summary `reports/eval_report.md`. Seed everything (`seed=42`). Cache LLM outputs in `data/eval/*.jsonl` (temperature 0) so reruns are free and deterministic.

### 12.1 Data (`scripts/build_eval_data.py`)
- Source: SQuAD v1.1 validation split (`datasets.load_dataset("rajpurkar/squad", split="validation")`).
- Sample 300 questions (seed 42). Gold document = the question's `context`. Corpus = gold contexts ∪ 700 other random contexts, deduplicated by text; `doc_id = sha1(text)[:12]`; `source="squad"`. Write `data/corpus.jsonl`, `data/eval/questions_clean.jsonl` (`question`, `gold_doc_id`, `gold_answers`).
- `degraded` slice: for each question, one LLM call producing a vague version (prompt: `Rewrite the question to be vague: remove proper nouns and specific numbers, use generic references, keep the same information need. Output the question only.`). Write `questions_degraded.jsonl`. Same gold ids.

### 12.2 Retrieval experiments (`scripts/eval_retrieval.py`)
For both slices compute: Recall@5 and MRR@5 for (a) dense-only, (b) hybrid RRF top 5, (c) hybrid + cross-encoder single pass; retrieval error rate with `max_rewrites=0` vs `max_rewrites=2` (run the compiled graph with the real deps, skipping generation by using a deps wrapper whose `generate` returns `""`); `rewrite_rate` (share of queries that triggered ≥1 rewrite); histogram of first-pass `relevance_score` (10 bins). If `rewrite_rate` is ≈0% or ≈100%, state it in the report; do not change the 0.70 threshold.

### 12.3 Answer quality (`scripts/eval_ragas.py`)
- Run the full pipeline on the first 200 clean questions (concurrency semaphore = 4 to respect rate limits).
- Ragas: before writing code, run `python -c "import ragas; print(ragas.__version__)"` and read the installed API (`help`/`inspect`); isolate it in one adapter function `run_ragas(samples) -> dict`. Judge LLM = the Gemini model through LangChain wrapper. Metrics: `faithfulness`, `answer_relevancy`, and `answer_correctness` if it exists in the installed version.
- If the adapter fails after 3 repair attempts, record `ragas: "skipped"` with the error in `reports/blockers.md`, and still report `gold_answer_contained`.

### 12.4 Latency (`scripts/bench_latency.py`)
Warm up 10 requests, then 100 sequential `POST /v1/query` against a running server (or in-process ASGI with real deps). From each response's `step_log` sum durations of `retrieve`, `grade_documents`, `rewrite_query` → `retrieval_path_ms`; wall-clock → `end_to_end_ms`. Report p50/p95 for both, plus concurrency-8 throughput over 100 requests.

### 12.5 Report
`reports/eval_report.md` contains: the §1 table with a **Measured** column, hardware (CPU/GPU, RAM), model names/versions, dataset sizes, seeds, and a "Resume bullets to edit" list for every claim whose measured value differs from the resume.

---

## 13. Phased execution plan with gates

| Phase | Deliverable | Gate (all must exit 0) |
|---|---|---|
| P0 | venv, `requirements.txt`, lock file, layout, `pytest.ini`, `Makefile`, `DECISIONS.md` | `python -c "import langgraph, fastapi, qdrant_client, rank_bm25; print('ok')"` |
| P1 | `config.py`, `state.py`, `routing.py`, `graph.py`, fakes, routing/graph tests | `pytest -q tests/test_routing.py tests/test_graph.py` |
| P2 | `qdrant_store.py`, `deps.py` (Embedder, Bm25Index, CrossEncoderScorer, LLM wrappers, `RealDeps`), unit tests | `pytest -q` and `pytest -q -m network tests/test_cross_encoder_network.py` |
| P3 | `main.py`, `asgi.py`, API tests, forbidden-pattern test | `pytest -q`; `uvicorn app.asgi:app` boots and `curl -sf localhost:8000/healthz` returns `{"ok":true}` |
| P4 | `ingest.py`, `build_eval_data.py`; Qdrant via docker compose; corpus ingested | `python scripts/ingest.py` prints matching counts; `curl -s localhost:8000/readyz` lists the collection |
| P5 | `eval_retrieval.py`, `eval_ragas.py`, `bench_latency.py`, `reports/eval_report.md` | the three scripts exit 0 and the JSON reports exist |
| P6 | `README.md` (architecture diagram, run/eval instructions, measured results table), `reports/SELF_AUDIT.md` | §15 checklist complete |

`Makefile` targets: `install`, `test` (`pytest -q`), `up` (docker compose), `ingest`, `run` (`uvicorn app.asgi:app --port 8000`), `eval`, `bench`, `check` (`test` + forbidden-pattern test + `eval` summary present).

---

## 14. Known pitfalls

1. Cross-encoder logits are not probabilities; apply sigmoid exactly once (§6.2).
2. Qdrant ids: only uint or UUID (§7.1-7).
3. `client.search` is deprecated → `query_points` returns an object; use `.points`.
4. Stateful fakes reused across requests produce misleading test results; instantiate per request.
5. Python < 3.11 breaks `RunnableConfig` propagation in async nodes.
6. `BM25Okapi` on empty/OOV tokens: guard empty token list; filter `score > 0`.
7. LangChain message `content` may be a list of blocks (§6.4).
8. Each rewrite loop costs 3 super-steps (retrieve, grade, rewrite); with `max_rewrites=2` the run stays under ~10 steps; keep `recursion_limit=25`.
9. First request is slow without warm-up (§6.5-3).
10. `pydantic-settings` treats `QDRANT_URL=` (empty) as `""`, not `None`; the validator in `config.py` handles it.
11. LLM rate limits during eval: semaphore + cached outputs.
12. Sentence-transformers / transformers downloads need network: keep them out of default tests.

---

## 15. Definition of done and self-audit

Produce `reports/SELF_AUDIT.md` containing, for each item, the evidence (command + output excerpt):

- [ ] `pytest -q` green; test count reported.
- [ ] `tests/test_forbidden_patterns.py` green (no sync Qdrant, no `on_event`, no blocking calls, no placeholders).
- [ ] `decide_to_generate` boundary table verified (0.699999 → rewrite, 0.70 → generate, `None` → rewrite, exhausted budget → generate).
- [ ] Loop terminates in every test scenario; `step_log` grows across passes.
- [ ] `grep -rn "AsyncQdrantClient" app/` shows the single construction site in `main.py`; `grep -rn "to_thread" app/` shows every blocking call wrapped.
- [ ] Server boots; `/healthz`, `/readyz`, `/v1/query`, `/v1/query/stream` exercised with real deps against dockerised Qdrant.
- [ ] §1 table filled with measured values; resume-bullet edits listed.
- [ ] No secret in git history or files; `.env` ignored.
- [ ] `requirements.lock.txt` committed; versions recorded in `DECISIONS.md`.
