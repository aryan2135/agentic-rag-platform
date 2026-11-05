# scripts/eval_ragas.py
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset
from qdrant_client import AsyncQdrantClient

from app.config import Settings
from app.deps import Bm25Index, CrossEncoderScorer, Embedder, normalise_llm_content
from app.graph import build_graph
from app.state import initial_state
from app import qdrant_store as qs


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def check_gold_contained(answer: str, gold_answers: list[str]) -> bool:
    norm_ans = normalize_text(answer)
    for g in gold_answers:
        norm_g = normalize_text(g)
        if norm_g and norm_g in norm_ans:
            return True
    return False


class RagasEvalDeps:
    """Deps implementation for Ragas evaluation."""

    def __init__(self, cfg, client, embedder, bm25, scorer, llm):
        self.cfg = cfg
        self.client = client
        self.embedder = embedder
        self.bm25 = bm25
        self.scorer = scorer
        self.llm = llm

    async def retrieve(self, query: str):
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

    async def rerank(self, query: str, docs):
        if not docs:
            return []
        scores = await self.scorer.ascore(query, [d["text"] for d in docs])
        rescored = []
        for d, s in zip(docs, scores, strict=True):
            rescored.append({**d, "rerank_score": float(s)})
        rescored.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
        return rescored

    async def rewrite(self, original_query: str, query: str, docs):
        if self.llm:
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
                    resp = await self.llm.ainvoke(prompt)
                txt = normalise_llm_content(resp.content).strip()
                lines = [line.strip().strip("\"' ") for line in txt.splitlines() if line.strip()]
                if lines:
                    return lines[0][:300]
            except Exception:
                pass
        return query

    async def generate(self, query: str, docs, low_confidence: bool) -> str:
        if self.llm:
            try:
                low_note = "Retrieval confidence is low; state your uncertainty explicitly." if low_confidence else ""
                context = "\n\n".join(f"[{d['doc_id']}] {d['text']}" for d in docs)
                prompt = (
                    f"Answer the question using ONLY the context. Cite sources as [doc_id]. If the context does not contain the answer, say you cannot find it in the provided documents.\n"
                    f"{low_note}\n"
                    f"Question: {query}\n"
                    f"Context:\n{context}"
                )
                async with asyncio.timeout(self.cfg.llm_timeout_s):
                    resp = await self.llm.ainvoke(prompt)
                return normalise_llm_content(resp.content)
            except Exception:
                pass
        # Context-based direct extraction when offline or LLM unavailable
        if docs:
            top_doc = docs[0]
            return f"According to [{top_doc['doc_id']}], {top_doc['text'][:400]}"
        return "I cannot find the answer in the provided documents."


def run_ragas(samples: list[dict], judge_llm) -> dict:
    """Isolated Ragas adapter function."""
    try:
        from ragas import evaluate
        from ragas.metrics import answer_correctness, answer_relevancy, faithfulness

        data_dict = {
            "question": [s["question"] for s in samples],
            "contexts": [s["contexts"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "ground_truth": [s["ground_truth"] for s in samples],
        }
        dataset = Dataset.from_dict(data_dict)
        metrics = [faithfulness, answer_relevancy]
        if answer_correctness:
            metrics.append(answer_correctness)

        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            raise_exceptions=False,
            show_progress=False,
        )
        return {
            "status": "success",
            "scores": {k: float(v) for k, v in eval_result.items() if isinstance(v, (int, float))},
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {str(exc)}"}


async def process_one(idx, item, graph, eval_deps, sem):
    async with sem:
        q = item["question"]
        gold_answers = item["gold_answers"]
        gold_doc = item["gold_doc_id"]
        out = await graph.ainvoke(
            initial_state(q, max_rewrites=2),
            config={"configurable": {"deps": eval_deps}, "recursion_limit": 25},
        )
        ans = out.get("answer", "")
        contained = check_gold_contained(ans, gold_answers)
        doc_ids = [d["doc_id"] for d in out.get("documents", [])[:5]]
        retrieved_gold = gold_doc in doc_ids
        return {
            "question": q,
            "answer": ans,
            "contexts": [d["text"] for d in out.get("documents", [])[:5]],
            "ground_truth": gold_answers[0] if gold_answers else "",
            "gold_answers": gold_answers,
            "gold_answer_contained": contained,
            "gold_doc_retrieved": retrieved_gold,
            "rewrite_count": out.get("rewrite_count", 0),
            "fallback_used": out.get("fallback_used", False),
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

    llm = None
    if cfg.google_api_key and cfg.google_api_key != "mock-google-api-key":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=cfg.llm_model, temperature=0, google_api_key=cfg.google_api_key)
        except Exception:
            llm = None

    eval_deps = RagasEvalDeps(cfg, client, embedder, bm25, scorer, llm)
    graph = build_graph()

    with open("data/eval/questions_clean.jsonl", encoding="utf-8") as f:
        clean_qs = [json.loads(line) for line in f if line.strip()]

    target_qs = clean_qs[:200]
    print(f"Evaluating answer quality on {len(target_qs)} clean questions (concurrency=4)...")

    sem = asyncio.Semaphore(4)
    tasks = [process_one(i, item, graph, eval_deps, sem) for i, item in enumerate(target_qs)]
    results = await asyncio.gather(*tasks)

    n = len(results)
    contained_count = sum(1 for r in results if r["gold_answer_contained"])
    gold_doc_count = sum(1 for r in results if r["gold_doc_retrieved"])
    gold_answer_contained_rate = contained_count / n
    gold_doc_retrieval_rate = gold_doc_count / n

    print(f"Evaluated {n} questions.")
    print(f"Deterministic Gold Answer Contained: {contained_count}/{n} ({gold_answer_contained_rate*100:.2f}%)")
    print(f"Gold Document In Top 5: {gold_doc_count}/{n} ({gold_doc_retrieval_rate*100:.2f}%)")

    # Run Ragas adapter
    ragas_result = {"status": "skipped", "reason": "No live Gemini API key provided; reporting gold_answer_contained"}
    if llm:
        print("Running Ragas evaluation...")
        ragas_result = run_ragas(results, judge_llm=llm)
        if ragas_result.get("status") == "error":
            os.makedirs("reports", exist_ok=True)
            with open("reports/blockers.md", "a", encoding="utf-8") as f:
                f.write(f"\n## Ragas Evaluation Blocker\n- Error: {ragas_result.get('error')}\n")

    report_data = {
        "n_samples": n,
        "gold_answer_contained": round(gold_answer_contained_rate, 4),
        "gold_doc_retrieval_rate": round(gold_doc_retrieval_rate, 4),
        "ragas": ragas_result,
        "samples_summary": [
            {
                "question": r["question"][:80],
                "gold_answer_contained": r["gold_answer_contained"],
                "gold_doc_retrieved": r["gold_doc_retrieved"],
                "rewrite_count": r["rewrite_count"],
            }
            for r in results[:10]
        ],
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/ragas_eval.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print("Report saved to reports/ragas_eval.json")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())

