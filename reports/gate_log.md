# Phase Gate Log

## Phase 0: Environment & Layout
- **Command**: `python -c "import langgraph, fastapi, qdrant_client, rank_bm25; print('ok')"`
- **Exit Code**: 0
- **Output**:
```
ok
```
- **Status**: PASSED


## Phase 1: Core State, Routing & Graph Loop
- **Command**: `pytest -q tests/test_routing.py tests/test_graph.py`
- **Exit Code**: 0
- **Output**:
```
............                                                             [100%]
12 passed in 0.73s
```
- **Status**: PASSED

## Phase 2: Qdrant Store, Real Dependencies & Scorer
- **Command 1**: `pytest -q`
- **Exit Code**: 0
- **Output**:
```
s.....................                                                   [100%]
21 passed, 1 skipped in 13.70s
```
- **Command 2**: `pytest -q -m network tests/test_cross_encoder_network.py`
- **Exit Code**: 0
- **Output**:
```
.                                                                        [100%]
1 passed, 2 warnings in 46.82s
```
- **Status**: PASSED

## Phase 3: FastAPI Application & Endpoints
- **Command 1**: `pytest -q`
- **Exit Code**: 0
- **Output**:
```
..s......................                                                [100%]
24 passed, 1 skipped in 36.18s
```
- **Command 2**: `uvicorn app.asgi:app` boot and `curl -sf http://localhost:8000/healthz`
- **Exit Code**: 0
- **Output**:
```json
{"ok":true}
```
- **Status**: PASSED

## Phase 4: Ingestion & Qdrant Integration
- **Command 1**: `python scripts/ingest.py`
- **Exit Code**: 0
- **Output**:
```
Loaded 983 documents from data/corpus.jsonl
Upserted batch 1/16 (64 docs)
Upserted batch 2/16 (64 docs)
Upserted batch 3/16 (64 docs)
Upserted batch 4/16 (64 docs)
Upserted batch 5/16 (64 docs)
Upserted batch 6/16 (64 docs)
Upserted batch 7/16 (64 docs)
Upserted batch 8/16 (64 docs)
Upserted batch 9/16 (64 docs)
Upserted batch 10/16 (64 docs)
Upserted batch 11/16 (64 docs)
Upserted batch 12/16 (64 docs)
Upserted batch 13/16 (64 docs)
Upserted batch 14/16 (64 docs)
Upserted batch 15/16 (64 docs)
Upserted batch 16/16 (23 docs)
Wrote BM25 corpus to data/bm25_corpus.jsonl
Qdrant collection 'docs' has 983 points (expected 983)
Ingestion verified successfully!
```
- **Command 2**: `uvicorn app.asgi:app` boot and `curl -s http://localhost:8000/readyz`
- **Exit Code**: 0
- **Output**:
```json
{"ok":true,"collections":["docs"]}
```
- **Status**: PASSED

## Phase 5: Evaluation Harness & Benchmarking
- **Command 1**: `python scripts/eval_retrieval.py`
- **Exit Code**: 0
- **Artifact**: `reports/retrieval_eval.json` (created)
- **Command 2**: `python scripts/eval_ragas.py`
- **Exit Code**: 0
- **Artifact**: `reports/ragas_eval.json` (created)
- **Command 3**: `python scripts/bench_latency.py`
- **Exit Code**: 0
- **Artifact**: `reports/latency_bench.json` (created)
- **Summary Report**: `reports/eval_report.md` (created)
- **Status**: PASSED

## Phase 6: Documentation & Self-Audit
- **Artifact 1**: `README.md` (architecture diagram, pipeline docs, run/eval instructions, measured claims)
- **Artifact 2**: `reports/SELF_AUDIT.md` (all 9 checklist items verified with command evidence)
- **Status**: PASSED
