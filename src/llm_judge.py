"""[WRITE-NEW] LLM-as-judge (GPT), provider-agnostic interface.

judge(student, professor) -> 0/1/2 relevance. Reads OPENAI_API_KEY from env.
Kept separate from explain.py so the judge never grades its own generated text.
"""
# TODO: llm_judge(student, professor) -> int
