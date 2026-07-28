"""End-to-end single-student demo: query -> Top-5 professors + grounded RAG explanations.

Reuses the SAVED paper embeddings (outputs/paper_embs_<encoder>.npy) so it only encodes
one query — fast enough for a live demo. Prints (and returns) the recommendations.

Usage (local, needs OPENAI_API_KEY for explanations):
    python src/recommend_demo.py                      # random sample student
    python src/recommend_demo.py NU_ENG_000042        # a specific student
    python src/recommend_demo.py --text "I want to work on soft robotics and control"
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_students, load_professors, load_papers
from retrieve import build_student_query, recommend_one
from embed import encode
from explain import explain


def run(student_id=None, free_text=None, encoder="minilm", k=5, with_explanations=True):
    students, profs, papers = load_students(), load_professors(), load_papers()
    paper_embs = np.load(ROOT / "outputs" / f"paper_embs_{encoder}.npy")

    if free_text:
        query, who = free_text, "(free text)"
    else:
        if student_id:
            match = students[students["student_id"] == student_id]
            if match.empty:
                print(f"[!] student_id '{student_id}' not found. "
                      f"Try one like: {students['student_id'].iloc[0]}")
                return []
            row = match.iloc[0]
        else:
            row = students.sample(1).iloc[0]
        query, who = build_student_query(row), f'{row["student_id"]} ({row["major"]})'

    q_emb = encode([query], encoder)[0]
    top = recommend_one(q_emb, paper_embs, papers, k)

    print(f"\nStudent: {who}\nQuery: {query[:160]}…\n" + "=" * 60)
    results = []
    for name, score, paper in top:
        line = f"{score:.3f}  {name}   (via: {str(paper['Paper Title'])[:60]})"
        print("\n" + line)
        if with_explanations:
            reason = explain(query, name, [paper])
            print("  → " + reason)
        results.append((name, score, paper, None))
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--text":
        run(free_text=" ".join(args[1:]))
    elif args:
        run(student_id=args[0])
    else:
        run()