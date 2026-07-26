"""[WRITE-NEW] RAG explanation via GPT. Reads OPENAI_API_KEY from env (never hardcode).

Given (student, professor, hit_papers) -> short 'why recommended' grounded ONLY in the
retrieved papers (cite titles). No free-form claims beyond the retrieved evidence.
"""
# TODO: explain(student, professor, hit_papers) -> str
