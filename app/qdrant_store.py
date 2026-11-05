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

