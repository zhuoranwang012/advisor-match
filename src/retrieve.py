"""[WRITE-NEW] Paper-level retrieval, then aggregate to professors.

student query -> cosine vs paper vectors -> top papers
             -> max-pool paper scores to each author (professor) -> Top-K profs.
Return, per recommended professor, the specific papers that triggered the match
(these feed the RAG explanation).
"""
# TODO: recommend(student_text, k=5) -> [(professor, score, hit_papers)]
