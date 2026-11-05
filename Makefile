.PHONY: install test up ingest run eval bench check

install:
	uv pip install -r requirements.txt --python .venv

test:
	pytest -q

up:
	docker compose up -d

ingest:
	python scripts/ingest.py

run:
	uvicorn app.asgi:app --port 8000

eval:
	python scripts/eval_retrieval.py
	python scripts/eval_ragas.py

bench:
	python scripts/bench_latency.py

check:
	pytest -q
	pytest -q tests/test_forbidden_patterns.py

