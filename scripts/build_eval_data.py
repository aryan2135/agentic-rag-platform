# scripts/build_eval_data.py
import hashlib
import json
import os
import random
import re
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset


def doc_id_for(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def fallback_degrade_question(q: str) -> str:
    # Rule-based vague rewrite when LLM is unavailable:
    # Remove numbers/years, replace specific capitalized phrases with generic references
    text = re.sub(r"\b(19\d\d|20\d\d|\d+)\b", "a certain time", q)
    words = text.split()
    degraded_words = []
    for w in words:
        if w.istitle() and not w.lower() in {"what", "when", "where", "which", "who", "why", "how"}:
            degraded_words.append("this entity")
        else:
            degraded_words.append(w)
    degraded = " ".join(degraded_words)
    degraded = re.sub(r"(this entity\s*)+", "the relevant entity ", degraded)
    if not degraded.endswith("?"):
        degraded += "?"
    return degraded


def main():
    print("Loading SQuAD validation dataset...")
    ds = load_dataset("rajpurkar/squad", split="validation")

    total_examples = len(ds)
    print(f"Total SQuAD validation items: {total_examples}")

    rng = random.Random(42)
    indices = list(range(total_examples))
    rng.shuffle(indices)

    sample_indices = indices[:300]
    sampled_examples = [ds[i] for i in sample_indices]

    gold_contexts = {}
    clean_questions = []

    for ex in sampled_examples:
        ctx = ex["context"]
        d_id = doc_id_for(ctx)
        gold_contexts[d_id] = ctx
        clean_questions.append({
            "question": ex["question"],
            "gold_doc_id": d_id,
            "gold_answers": ex["answers"]["text"],
        })

    # Collect 700 other random contexts not already in gold_contexts
    other_indices = indices[300:]
    other_contexts = {}
    for i in other_indices:
        ctx = ds[i]["context"]
        d_id = doc_id_for(ctx)
        if d_id not in gold_contexts and d_id not in other_contexts:
            other_contexts[d_id] = ctx
        if len(other_contexts) >= 700:
            break

    print(f"Gold contexts: {len(gold_contexts)}, Other contexts: {len(other_contexts)}")

    corpus = []
    for d_id, ctx in gold_contexts.items():
        corpus.append({"doc_id": d_id, "text": ctx, "source": "squad"})
    for d_id, ctx in other_contexts.items():
        corpus.append({"doc_id": d_id, "text": ctx, "source": "squad"})

    os.makedirs("data", exist_ok=True)
    os.makedirs("data/eval", exist_ok=True)

    with open("data/corpus.jsonl", "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(corpus)} documents to data/corpus.jsonl")

    with open("data/eval/questions_clean.jsonl", "w", encoding="utf-8") as f:
        for q in clean_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote {len(clean_questions)} questions to data/eval/questions_clean.jsonl")

    # Generate degraded slice
    # Check if we can use LLM, else fallback to deterministic degradation
    from app.config import Settings
    cfg = Settings()

    llm = None
    if cfg.google_api_key and cfg.google_api_key != "mock-google-api-key":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=cfg.llm_model, temperature=0, google_api_key=cfg.google_api_key)
        except Exception:
            llm = None

    prompt_tmpl = "Rewrite the question to be vague: remove proper nouns and specific numbers, use generic references, keep the same information need. Output the question only.\nQuestion: {question}"

    degraded_questions = []
    print("Generating degraded questions...")
    for q in clean_questions:
        orig_q = q["question"]
        deg_q = None
        if llm:
            try:
                resp = llm.invoke(prompt_tmpl.format(question=orig_q))
                deg_q = resp.content.strip().splitlines()[0].strip("\"' ")
            except Exception:
                deg_q = None
        if not deg_q:
            deg_q = fallback_degrade_question(orig_q)

        degraded_questions.append({
            "question": deg_q,
            "gold_doc_id": q["gold_doc_id"],
            "gold_answers": q["gold_answers"],
        })

    with open("data/eval/questions_degraded.jsonl", "w", encoding="utf-8") as f:
        for q in degraded_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote {len(degraded_questions)} questions to data/eval/questions_degraded.jsonl")


if __name__ == "__main__":
    main()
