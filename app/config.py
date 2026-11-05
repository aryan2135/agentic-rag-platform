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

