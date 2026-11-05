# app/deps.py
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Sequence

from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
from qdrant_client import AsyncQdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import Settings
from app import qdrant_store as qs
from app.state import RetrievedDoc

REWRITE_PROMPT: str = """You rewrite search queries for a retrieval system.
Original question: {original}
Current query: {current}
Top retrieved passages (low relevance):
{snippets}
Write ONE improved search query: more specific, keyword-rich, same intent. Output the query only. No quotes, no explanation."""

GENERATE_PROMPT: str = """Answer the question using ONLY the context. Cite sources as [doc_id]. If the context does not contain the answer, say you cannot find it in the provided documents.
{low_confidence_note}
Question: {question}
Context:
{context}"""

_TOKEN: re.Pattern[str] = re.compile(r"\w+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def normalise_llm_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(str(b.get("text", "")))
            elif getattr(b, "type", None) == "text":
                parts.append(str(getattr(b, "text", "")))
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return str(content)


class Bm25Index:
    def __init__(self, path: str) -> None:
        import os
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self._docs = [json.loads(line) for line in f if line.strip()]
        else:
            self._docs = []
        self._ids = [d["doc_id"] for d in self._docs]
        self._by_id = {d["doc_id"]: d for d in self._docs}
        corpus_toks = [_tok(d["text"]) for d in self._docs if _tok(d["text"])]
        if corpus_toks:
            self._bm25 = BM25Okapi(corpus_toks)
        else:
            self._bm25 = None

    def payload(self, doc_id: str) -> dict[str, Any]:
        return self._by_id[doc_id]

    def _top_k(self, query: str, k: int) -> list[tuple[str, float]]:
        toks = _tok(query)
        if not toks or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(toks)
        idx = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i])) for i in idx if scores[i] > 0]

    async def atop_k(self, query: str, k: int) -> list[tuple[str, float]]:
        return await asyncio.to_thread(self._top_k, query, k)


class Embedder:
    QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension()

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embs.tolist()

    async def aencode_docs(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode_sync, texts)

    async def aencode_query(self, query: str) -> list[float]:
        prefixed = f"{self.QUERY_PREFIX}{query}"
        res = await asyncio.to_thread(self._encode_sync, [prefixed])
        return res[0]


class CrossEncoderScorer:
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        if self._model.config.num_labels != 1:
            raise ValueError("expected a single-logit relevance model")
        self._max_length = max_length

    @torch.inference_mode()
    def _score_sync(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        enc = self._tok(
            [query] * len(texts),
            texts,
            padding=True,
            truncation="only_second",
            max_length=self._max_length,
            return_tensors="pt",
        )
        logits = self._model(**enc).logits.squeeze(-1)
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        return torch.sigmoid(logits).tolist()

    async def ascore(self, query: str, texts: list[str]) -> list[float]:
        return await asyncio.to_thread(self._score_sync, query, texts)


class RealDeps:
    """Production implementation of the Deps protocol."""

    def __init__(
        self,
        cfg: Settings,
        client: AsyncQdrantClient,
        embedder: Embedder,
        bm25: Bm25Index,
        scorer: CrossEncoderScorer,
        llm: ChatGoogleGenerativeAI,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.embedder = embedder
        self.bm25 = bm25
        self.scorer = scorer
        self.llm = llm

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
        return [
            {
                "doc_id": d,
                "text": payloads[d]["text"],
                "source": payloads[d].get("source", ""),
                "rrf_score": s,
            }
            for d, s in top
        ]

    async def rerank(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not docs:
            return []
        scores = await self.scorer.ascore(query, [d["text"] for d in docs])
        rescored: list[RetrievedDoc] = []
        for d, s in zip(docs, scores, strict=True):
            entry: RetrievedDoc = {
                "doc_id": d["doc_id"],
                "text": d["text"],
                "source": d["source"],
                "rrf_score": d["rrf_score"],
                "rerank_score": float(s),
            }
            rescored.append(entry)
        rescored.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
        return rescored

    async def rewrite(self, original_query: str, query: str, docs: list[RetrievedDoc]) -> str:
        snippets = "\n".join(f"[{d['doc_id']}] {d['text']}" for d in docs[:5])
        prompt = REWRITE_PROMPT.format(original=original_query, current=query, snippets=snippets)
        try:
            async with asyncio.timeout(self.cfg.llm_timeout_s):
                resp = await self.llm.ainvoke(prompt)
            raw = normalise_llm_content(resp.content).strip()
            lines = [line.strip().strip("\"' \t\r\n") for line in raw.splitlines() if line.strip()]
            candidate = lines[0][:300] if lines else ""
            return candidate if candidate else query
        except Exception:
            return query

    async def generate(self, query: str, docs: list[RetrievedDoc], low_confidence: bool) -> str:
        low_confidence_note = (
            "Retrieval confidence is low; state your uncertainty explicitly."
            if low_confidence
            else ""
        )
        context = "\n\n".join(f"[{d['doc_id']}] {d['text']}" for d in docs)
        prompt = GENERATE_PROMPT.format(
            low_confidence_note=low_confidence_note,
            question=query,
            context=context,
        )
        try:
            async with asyncio.timeout(self.cfg.llm_timeout_s):
                resp = await self.llm.ainvoke(prompt)
            return normalise_llm_content(resp.content)
        except Exception:
            if docs:
                return f"According to [{docs[0]['doc_id']}], {docs[0]['text'][:300]}"
            return "I cannot find the answer in the provided documents."


async def build_real_deps(cfg: Settings, client: AsyncQdrantClient) -> RealDeps:
    if not cfg.google_api_key:
        raise ValueError("google_api_key is required to build RealDeps")

    embedder, scorer, bm25 = await asyncio.gather(
        asyncio.to_thread(Embedder, cfg.embed_model),
        asyncio.to_thread(CrossEncoderScorer, cfg.rerank_model),
        asyncio.to_thread(Bm25Index, cfg.bm25_corpus_path),
    )
    await qs.ensure_collection(client, cfg.qdrant_collection, embedder.dim)
    await embedder.aencode_query("warmup")
    await scorer.ascore("warmup", ["warmup document"])

    llm = ChatGoogleGenerativeAI(
        model=cfg.llm_model,
        temperature=0,
        google_api_key=cfg.google_api_key,
    )
    return RealDeps(cfg=cfg, client=client, embedder=embedder, bm25=bm25, scorer=scorer, llm=llm)

