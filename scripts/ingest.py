# scripts/ingest.py
import asyncio
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import AsyncQdrantClient

from app.config import Settings
from app.deps import Embedder
from app import qdrant_store as qs


async def main() -> None:
    cfg = Settings()
    corpus_path = "data/corpus.jsonl"
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"{corpus_path} not found. Run scripts/build_eval_data.py first.")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(corpus)} documents from {corpus_path}")

    # Use Qdrant instance from settings (defaults to localhost:6333 if set, or in-memory)
    client = (
        AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key, timeout=10)
        if cfg.qdrant_url
        else AsyncQdrantClient(":memory:")
    )

    try:
        embedder = Embedder(cfg.embed_model)
        await qs.ensure_collection(client, cfg.qdrant_collection, embedder.dim)

        batch_size = 64
        total = len(corpus)
        for i in range(0, total, batch_size):
            batch = corpus[i : i + batch_size]
            texts = [d["text"] for d in batch]
            doc_ids = [d["doc_id"] for d in batch]
            vectors = await embedder.aencode_docs(texts)
            await qs.upsert_batch(client, cfg.qdrant_collection, doc_ids, vectors, batch)
            print(f"Upserted batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} ({len(batch)} docs)")

        # Write data/bm25_corpus.jsonl (same records)
        bm25_path = cfg.bm25_corpus_path
        os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
        with open(bm25_path, "w", encoding="utf-8") as f:
            for d in corpus:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Wrote BM25 corpus to {bm25_path}")

        # Check counts
        count_res = await client.count(cfg.qdrant_collection, exact=True)
        print(f"Qdrant collection '{cfg.qdrant_collection}' has {count_res.count} points (expected {len(corpus)})")
        assert count_res.count == len(corpus), f"Mismatch: {count_res.count} != {len(corpus)}"
        print("Ingestion verified successfully!")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
