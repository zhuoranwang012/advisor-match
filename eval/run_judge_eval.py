"""Evaluate random / keyword / semantic under LLM-judged relevance (the PRIMARY eval).

Judges each unique (student, professor) pair once (cached), then scores every method
on the same judged labels. Small sample by default to control API cost.

Usage (local, needs OPENAI_API_KEY):
    python eval/run_judge_eval.py --semantic outputs/rankings_minilm.json --sample 60
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from data_loader import load_students, load_professors, load_papers
from baselines import student_words, prof_words, random_recommend, keyword_recommend
from llm_judge import judge_pair
from metrics import hit_at_k, precision_at_k, ndcg_at_k, mrr

K = 5


def score(rankings, relevance, ids):
    a = np.array([(hit_at_k(rankings[i], relevance[i], K),
                   precision_at_k(rankings[i], relevance[i], K),
                   ndcg_at_k(rankings[i], relevance[i], K),
                   mrr(rankings[i], relevance[i])) for i in ids])
    return dict(zip(["Hit@5", "Prec@5", "NDCG@5", "MRR"], a.mean(axis=0).round(3)))


def main(semantic_path, sample=60, seed=0, threshold=1):
    students, profs, papers = load_students(), load_professors(), load_papers()
    semantic = json.load(open(semantic_path))

    rng = np.random.default_rng(seed)
    ids = list(rng.choice([s for s in students["student_id"] if s in semantic],
                          size=sample, replace=False))

    sview = students.set_index("student_id")
    pview = profs.set_index("professor_name")
    prof_names = profs["professor_name"].tolist()
    pwords = [prof_words(r) for _, r in profs.iterrows()]

    # build each method's Top-5 per sampled student
    method_rank = {"random": {}, "keyword": {}, "semantic": {}}
    for sid in ids:
        row = sview.loc[sid]
        method_rank["random"][sid] = random_recommend(prof_names, K, rng)
        method_rank["keyword"][sid] = keyword_recommend(student_words(row), profs, pwords, K)
        method_rank["semantic"][sid] = semantic[sid][:K]

    # judge each unique (student, prof) once
    cache_path = ROOT / "outputs" / "judge_cache.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    total = 0
    for sid in ids:
        s_text = str(sview.loc[sid].get("interest_profile", ""))
        union = set().union(*(method_rank[m][sid] for m in method_rank))
        for name in union:
            key = f"{sid}|||{name}"
            if key in cache or name not in pview.index:
                continue
            cache[key] = judge_pair(s_text, str(pview.loc[name].get("prof_text", "")))
            total += 1
            if total % 20 == 0:
                json.dump(cache, open(cache_path, "w")); print(f"  judged {total} pairs…")
    json.dump(cache, open(cache_path, "w"))
    print(f"Judged {total} new pairs (cache: {len(cache)}).")

    # relevance sets from judged scores
    relevance = {}
    for sid in ids:
        relevance[sid] = {name for m in method_rank for name in method_rank[m][sid]
                          if cache.get(f"{sid}|||{name}", 0) >= threshold}

    table = pd.DataFrame({m: score(method_rank[m], relevance, ids) for m in method_rank}).T
    print(f"\n=== LLM-judge evaluation (n={sample}, relevant if score>={threshold}) ===")
    print(table.to_string())
    table.to_csv(ROOT / "outputs" / "eval_judge.csv")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic", required=True, help="rankings_*.json from retrieve.py")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--threshold", type=int, default=1, help="min score to count as relevant (1 or 2)")
    args = ap.parse_args()
    main(args.semantic, args.sample, threshold=args.threshold)