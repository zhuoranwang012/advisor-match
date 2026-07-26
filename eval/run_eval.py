"""Score baselines (and, when available, the semantic method) under the area labels.

Runs random + keyword baselines here (no model needed). To add the semantic method,
pass a precomputed {student_id: [ranked prof names]} from the local embedding run.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from data_loader import load_students, load_professors, load_papers
from baselines import student_words, prof_words, random_recommend, keyword_recommend
from proxy_labels import build_relevance
from metrics import hit_at_k, precision_at_k, ndcg_at_k, mrr

K = 5


def score(rankings, relevance, ids):
    rows = [(hit_at_k(rankings[i], relevance[i], K),
             precision_at_k(rankings[i], relevance[i], K),
             ndcg_at_k(rankings[i], relevance[i], K),
             mrr(rankings[i], relevance[i])) for i in ids]
    a = np.array(rows)
    return dict(zip(["Hit@5", "Prec@5", "NDCG@5", "MRR"], a.mean(axis=0).round(3)))


def main(semantic_rankings=None, sample=None, seed=0):
    students, profs, papers = load_students(), load_professors(), load_papers()
    relevance, eligible = build_relevance(students, profs, papers)

    rng = np.random.default_rng(seed)
    if sample:
        eligible = list(rng.choice(eligible, size=min(sample, len(eligible)), replace=False))
    print(f"Eligible students (area-labelled): {len(eligible)}")

    sview = students.set_index("student_id")
    prof_names = profs["professor_name"].tolist()
    pwords = [prof_words(r) for _, r in profs.iterrows()]

    rand_rank, kw_rank = {}, {}
    for sid in eligible:
        row = sview.loc[sid]
        rand_rank[sid] = random_recommend(prof_names, K, rng)
        kw_rank[sid] = keyword_recommend(student_words(row), profs, pwords, K)

    results = {"random": score(rand_rank, relevance, eligible),
               "keyword": score(kw_rank, relevance, eligible)}
    if semantic_rankings:
        results["semantic"] = score(semantic_rankings, relevance, eligible)

    table = pd.DataFrame(results).T
    print("\n=== Area-coverage evaluation (proxy label) ===")
    print(table.to_string())
    table.to_csv(ROOT / "outputs" / "eval_area.csv")
    return table


if __name__ == "__main__":
    main()
