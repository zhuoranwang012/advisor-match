"""Paper-level retrieval, then aggregate to professors.

Pipeline:
  student query text -> encode -> cosine vs PAPER vectors -> top papers
                     -> max-pool paper scores to each author (professor) -> Top-K profs
Each recommended professor carries the specific papers that triggered the match
(these feed the RAG explanation in explain.py).

RUN LOCALLY (uses src/embed.py, which downloads a model). The retrieval/aggregation
logic here is sandbox-tested with mock vectors; only the encoding step needs a model.
"""
import numpy as np
import pandas as pd

from data_loader import load_students, load_professors, load_papers


def build_student_query(row):
    """Natural-language-leaning query (interest_profile first)."""
    parts = [
        str(row.get("interest_profile", "")),
        "Research interests: " + "; ".join(row.get("research_interests", [])),
        "Methods/tools: " + "; ".join(row.get("methods_tools", [])),
        "Keywords: " + "; ".join(row.get("profile_keywords", [])),
    ]
    return " ".join(p for p in parts if p and not p.endswith(": "))


def paper_text(papers_df):
    """Text to encode per paper: title + abstract."""
    return (papers_df["Paper Title"].fillna("") + ". " +
            papers_df["Abstract"].fillna("")).tolist()


def aggregate_to_professors(paper_scores, authors, k=5):
    """max-pool paper scores to each author -> Top-K (professor, score, best_paper_idx)."""
    best = {}
    for idx, (score, author) in enumerate(zip(paper_scores, authors)):
        if author not in best or score > best[author][0]:
            best[author] = (score, idx)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:k]
    return [(name, float(s), pidx) for name, (s, pidx) in ranked]


def recommend_all(student_embs, paper_embs, papers_df, student_ids, k=5):
    """Return {student_id: [ranked professor names]} for evaluation."""
    authors = papers_df["Professor Name (Original)"].tolist()
    sims = student_embs @ paper_embs.T                    # (n_students, n_papers)
    rankings = {}
    for i, sid in enumerate(student_ids):
        top = aggregate_to_professors(sims[i], authors, k)
        rankings[sid] = [name for name, _, _ in top]
    return rankings


def recommend_one(student_emb, paper_embs, papers_df, k=5):
    """Detailed Top-K for one student: professor + score + the matching paper row."""
    authors = papers_df["Professor Name (Original)"].tolist()
    sims = student_emb @ paper_embs.T
    top = aggregate_to_professors(sims, authors, k)
    return [(name, score, papers_df.iloc[pidx]) for name, score, pidx in top]


def run(encoder="minilm", sample=None, seed=0, save=True):
    """Full local run: encode papers + students, produce rankings, save to outputs/."""
    from pathlib import Path
    from embed import encode
    ROOT = Path(__file__).resolve().parents[1]

    students, profs, papers = load_students(), load_professors(), load_papers()
    if sample:
        students = students.sample(sample, random_state=seed).reset_index(drop=True)

    print(f"Encoding {len(papers)} papers with {encoder}…")
    paper_embs = encode(paper_text(papers), encoder)
    students["query"] = students.apply(build_student_query, axis=1)
    print(f"Encoding {len(students)} student queries…")
    student_embs = encode(students["query"].tolist(), encoder)

    ids = students["student_id"].tolist()
    rankings = recommend_all(student_embs, paper_embs, papers, ids)

    if save:
        import json
        out = ROOT / "outputs" / f"rankings_{encoder}.json"
        out.write_text(json.dumps(rankings))
        np.save(ROOT / "outputs" / f"paper_embs_{encoder}.npy", paper_embs)
        print(f"Saved rankings -> {out}")
    return rankings


if __name__ == "__main__":
    import sys
    enc = sys.argv[1] if len(sys.argv) > 1 else "minilm"
    run(encoder=enc)
