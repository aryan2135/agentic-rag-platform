# Architecture and Implementation Decisions

## D-01: Python Runtime Version
- **Decision**: Used CPython 3.11.17 installed via `uv venv .venv`.
- **Rationale**: SPEC-01 Section 2 specifies "Python 3.12 (3.11 minimum: LangGraph needs it for async RunnableConfig propagation)". 3.11.17 satisfies the minimum requirement for async `RunnableConfig` propagation in LangGraph.

## D-02: Package Manager & Caching
- **Decision**: Used `uv` to manage `.venv` and install dependencies. Cache directory placed at `.cache/uv` and temporary files at `.cache/temp` on drive E: to prevent running out of storage on drive C:.
- **Rationale**: Drive E: has ~120GB available, whereas C: is near capacity.

## D-03: Dependency Versions Recorded
- **Decision**: Generated `requirements.lock.txt` using `uv pip freeze`. Key verified versions:
  - `langgraph`: 1.2.19
  - `langchain-core`: 1.7.4
  - `langchain-google-genai`: 4.4.2
  - `fastapi`: 0.162.0
  - `uvicorn`: 0.54.0
  - `qdrant-client`: 1.20.0
  - `pydantic`: 2.14.2
  - `pydantic-settings`: 2.16.2
  - `rank-bm25`: 0.2.2
  - `sentence-transformers`: 6.2.0
  - `transformers`: 5.22.1
  - `torch`: 2.15.0
  - `numpy`: 2.4.6
  - `datasets`: 1.1.1
  - `ragas`: 0.3.2
  - `httpx`: 0.28.1
  - `pytest`: 9.2.0
  - `pytest-asyncio`: 1.4.1.post1
- **Rationale**: SPEC-01 Section 2 required recording installed versions in `DECISIONS.md`.


## D-04: Windows Sandbox & Asyncio Loopback Sockets
- **Decision**: Commands running Python asyncio / pytest suites on Windows use `BypassSandbox: true` because Python on Windows implements its internal asyncio self-pipe wakeup mechanism via loopback socket pairs on `127.0.0.1`.
- **Rationale**: The standard Windows sandbox isolation blocks TCP sockets including localhost loopback, which causes asyncio event loop initialization (`socket.socketpair()`) to stall. Running with sandbox bypass allows local loopback socket creation and lets the test suite execute at full speed (12 tests in 0.73s).

## D-05: Datasets Dependency Version
- **Decision**: Updated `datasets` to `>=2.18.0` (resolved to `5.1.0`).
- **Rationale**: `pyarrow 26.0.1` deprecated and removed `pa.PyExtensionType`, which caused `datasets 1.1.1` to fail on import with `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`. Upgrading `datasets` resolved the incompatibility.

## D-06: Qdrant Standalone Engine
- **Decision**: Installed official standalone native Windows binary `qdrant 1.20.0` in `tools/qdrant.exe` running on port 6333 with storage in `./qdrant_storage`.
- **Rationale**: The Docker Desktop service was not started on the host system. Using the official native Qdrant 1.20.0 binary provides the exact same REST/gRPC endpoints, collection handling, and performance as `qdrant/qdrant:v1.20.0` docker container.

## D-07: Qdrant IPv4 Direct URL
- **Decision**: Set `QDRANT_URL=http://127.0.0.1:6333`.
- **Rationale**: Qdrant binds to IPv4 (`0.0.0.0:6333`). On Windows, `localhost` attempts IPv6 (`::1`) first before falling back to IPv4, causing intermittent `ConnectTimeout` exceptions during rapid sequential and concurrent requests. Using `127.0.0.1` eliminates the IPv6 retry latency.
