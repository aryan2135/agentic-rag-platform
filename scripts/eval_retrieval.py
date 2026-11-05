# scripts/eval_retrieval.py
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import AsyncQdrantClient

from app.config import Settings
from app.deps import Bm25Index, CrossEncoderScorer, Embedder
from app.graph import build_graph
from app import qdrant_store as qs
from app.state import RetrievedDoc, initial_state


class EvalDeps:
    """Wrapper that runs real retrieval, grading, and query rewriting, but skips generation for fast eval."""

    def __init__(self, cfg, client, embedder, bm25, scorer, real_llm=None):
        self.cfg = cfg
        self.client = client
        self.embedder = embedder
        self.bm25 = bm25
        self.scorer = scorer
        self.real_llm = real_llm

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
        rescored = []
        for d, s in zip(docs, scores, strict=True):
            entry = dict(d)
            entry["rerank_score"] = float(s)
            rescored.append(entry)
        rescored.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
        return rescored

    async def rewrite(self, original_query: str, query: str, docs: list[RetrievedDoc]) -> str:
        # If real LLM is available, use it; otherwise fallback to extracting top keywords from documents
        if self.real_llm:
            try:
                snippets = "\n".join(f"[{d['doc_id']}] {d['text']}" for d in docs[:5])
                prompt = (
                    f"You rewrite search queries for a retrieval system.\n"
                    f"Original question: {original_query}\n"
                    f"Current query: {query}\n"
                    f"Top retrieved passages (low relevance):\n{snippets}\n"
                    f"Write ONE improved search query: more specific, keyword-rich, same intent. Output the query only. No quotes, no explanation."
                )
                async with asyncio.timeout(self.cfg.llm_timeout_s):
                    resp = await self.real_llm.ainvoke(prompt)
                txt = resp.content if isinstance(resp.content, str) else str(resp.content)
                lines = [line.strip().strip("\"' ") for line in txt.splitlines() if line.strip()]
                if lines:
                    return lines[0][:300]
            except Exception:
                pass
        # Fallback query expansion: append distinct keywords from top documents
        words = [w for d in docs[:3] for w in d["text"].split() if len(w) > 4 and w.isalpha()]
        extra = " ".join(words[:4])
        return f"{query} {extra}".strip()

    async def generate(self, query: str, docs: list[RetrievedDoc], low_confidence: bool) -> str:
        return ""


def compute_histogram(scores: list[float], bins: int = 10) -> dict[str, int]:
    counts = {f"[{i/10:.1f}, {(i+1)/10:.1f})": 0 for i in range(bins)}
    for s in scores:
        idx = min(int(s * 10), bins - 1)
        key = f"[{idx/10:.1f}, {(idx+1)/10:.1f})"
        counts[key] += 1
    return counts


async def eval_slice(name: str, questions: list[dict], cfg: Settings, client: AsyncQdrantClient, embedder: Embedder, bm25: Bm25Index, scorer: CrossEncoderScorer, graph):
    print(f"\n--- Evaluating slice: {name} ({len(questions)} queries) ---")
    eval_deps = EvalDeps(cfg, client, embedder, bm25, scorer)

    dense_recalls, dense_mrrs = [], []
    rrf_recalls, rrf_mrrs = [], []
    full_recalls, full_mrrs = [], []
    first_pass_scores = []
    err_single_count = 0
    err_loop_count = 0
    rewrites_triggered = 0

    for idx, item in enumerate(questions):
        q = item["question"]
        gold_id = item["gold_doc_id"]

        # 1. Dense only top 5
        qvec = await embedder.aencode_query(q)
        dense_pts = await qs.dense_search(client, cfg.qdrant_collection, qvec, 5)
        dense_ids = [p.payload["doc_id"] for p in dense_pts]
        dense_r = 1.0 if gold_id in dense_ids else 0.0
        dense_mrr = 1.0 / (dense_ids.index(gold_id) + 1) if gold_id in dense_ids else 0.0
        dense_recalls.append(dense_r)
        dense_mrrs.append(dense_mrr)

        # 2. Hybrid RRF top 5
        dense_top50, bm25_top50 = await asyncio.gather(
            qs.dense_search(client, cfg.qdrant_collection, qvec, 50),
            bm25.atop_k(q, 50),
        )
        fused = qs.rrf([[p.payload["doc_id"] for p in dense_top50], [d for d, _ in bm25_top50]], k=60)
        rrf_top5 = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:5]
        rrf_ids = [d for d, _ in rrf_top5]
        rrf_r = 1.0 if gold_id in rrf_ids else 0.0
        rrf_mrr = 1.0 / (rrf_ids.index(gold_id) + 1) if gold_id in rrf_ids else 0.0
        rrf_recalls.append(rrf_r)
        rrf_mrrs.append(rrf_mrr)

        # 3. Hybrid + cross-encoder single pass (fused top 20 -> reranked top 5)
        fused_top20 = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:20]
        payloads = {p.payload["doc_id"]: p.payload for p in dense_top50}
        for d, _ in bm25_top50:
            payloads.setdefault(d, bm25.payload(d))
        docs_20 = [{"doc_id": d, "text": payloads[d]["text"]} for d, _ in fused_top20]
        ranked_20 = await eval_deps.rerank(q, docs_20)
        kept_5 = ranked_20[:5]
        full_ids = [d["doc_id"] for d in kept_5]
        full_r = 1.0 if gold_id in full_ids else 0.0
        full_mrr = 1.0 / (full_ids.index(gold_id) + 1) if gold_id in full_ids else 0.0
        full_recalls.append(full_r)
        full_mrrs.append(full_mrr)

        first_score = sum(d["rerank_score"] for d in kept_5) / len(kept_5) if kept_5 else 0.0
        first_pass_scores.append(first_score)

        # 4. Corrective loop: max_rewrites=0 vs max_rewrites=2
        out0 = await graph.ainvoke(
            initial_state(q, max_rewrites=0),
            config={"configurable": {"deps": eval_deps}, "recursion_limit": 25},
        )
        out0_ids = [d["doc_id"] for d in out0["documents"][:5]]
        if gold_id not in out0_ids:
            err_single_count += 1

        out2 = await graph.ainvoke(
            initial_state(q, max_rewrites=2),
            config={"configurable": {"deps": eval_deps}, "recursion_limit": 25},
        )
        out2_ids = [d["doc_id"] for d in out2["documents"][:5]]
        if gold_id not in out2_ids:
            err_loop_count += 1

        if out2["rewrite_count"] > 0:
            rewrites_triggered += 1

        if (idx + 1) % 50 == 0 or idx == len(questions) - 1:
            print(f"Processed {idx + 1}/{len(questions)} queries...")

    n = len(questions)
    r_dense = sum(dense_recalls) / n
    mrr_dense = sum(dense_mrrs) / n
    r_rrf = sum(rrf_recalls) / n
    mrr_rrf = sum(rrf_mrrs) / n
    r_full = sum(full_recalls) / n
    mrr_full = sum(full_mrrs) / n

    recall_boost = (r_full - r_dense) / r_dense if r_dense > 0 else 0.0
    err_single = err_single_count / n
    err_loop = err_loop_count / n
    error_reduction = 1.0 - (err_loop / err_single) if err_single > 0 else 0.0
    rewrite_rate = rewrites_triggered / n
    hist = compute_histogram(first_pass_scores)

    return {
        "slice": name,
        "n_queries": n,
        "dense_only": {"recall_at_5": round(r_dense, 4), "mrr_at_5": round(mrr_dense, 4)},
        "hybrid_rrf": {"recall_at_5": round(r_rrf, 4), "mrr_at_5": round(mrr_rrf, 4)},
        "hybrid_rerank_single": {"recall_at_5": round(r_full, 4), "mrr_at_5": round(mrr_full, 4)},
        "recall_boost_rel": round(recall_boost, 4),
        "retrieval_error_single_pass": round(err_single, 4),
        "retrieval_error_corrective_loop": round(err_loop, 4),
        "error_reduction": round(error_reduction, 4),
        "rewrite_rate": round(rewrite_rate, 4),
        "relevance_score_histogram": hist,
    }


async def main():
    cfg = Settings()
    client = (
        AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key, timeout=10)
        if cfg.qdrant_url
        else AsyncQdrantClient(":memory:")
    )

    embedder = Embedder(cfg.embed_model)
    scorer = CrossEncoderScorer(cfg.rerank_model)
    bm25 = Bm25Index(cfg.bm25_corpus_path)
    graph = build_graph()

    with open("data/eval/questions_clean.jsonl", encoding="utf-8") as f:
        clean_qs = [json.loads(line) for line in f if line.strip()]

    with open("data/eval/questions_degraded.jsonl", encoding="utf-8") as f:
        degraded_qs = [json.loads(line) for line in f if line.strip()]

    try:
        res_clean = await eval_slice("clean", clean_qs, cfg, client, embedder, bm25, scorer, graph)
        res_degraded = await eval_slice("degraded", degraded_qs, cfg, client, embedder, bm25, scorer, graph)

        out_report = {
            "clean_slice": res_clean,
            "degraded_slice": res_degraded,
        }

        os.makedirs("reports", exist_ok=True)
        with open("reports/retrieval_eval.json", "w", encoding="utf-8") as f:
            json.dump(out_report, f, indent=2)

        print("\n=== Retrieval Evaluation Complete ===")
        print(f"Clean Slice: Dense Recall@5={res_clean['dense_only']['recall_at_5']}, Full Recall@5={res_clean['hybrid_rerank_single']['recall_at_5']}, Boost={res_clean['recall_boost_rel']*100:.1f}%, Error Reduction={res_clean['error_reduction']*100:.1f}%")
        print(f"Degraded Slice: Dense Recall@5={res_degraded['dense_only']['recall_at_5']}, Full Recall@5={res_degraded['hybrid_rerank_single']['recall_at_5']}, Boost={res_degraded['recall_boost_rel']*100:.1f}%, Error Reduction={res_degraded['error_reduction']*100:.1f}%")
        print("Saved to reports/retrieval_eval.json")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

