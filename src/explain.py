"""RAG explanation layer: generate a grounded 'why recommended' reason via GPT.

explain(student_text, professor_name, hit_papers) -> str

Grounding rules (enforced in the prompt):
- Base the reason ONLY on the provided paper title(s)/abstract(s) — the retrieval evidence.
- Cite at least one specific paper title.
- No claims beyond the retrieved papers; do NOT invent results the papers don't state.
- This keeps it a RAG explanation (retrieved papers = the 'R'), not free-form generation.

Reads OPENAI_API_KEY from the environment; never hardcode a key.
RUN LOCALLY (needs network + API key). Sandbox-checked for syntax only.
"""
import os
import time

MODEL = os.environ.get("EXPLAIN_MODEL", "gpt-4o-mini")

SYSTEM = (
    "You explain why a professor is a good research-advising match for a student. "
    "Use ONLY the provided paper titles and abstracts as evidence. Cite at least one "
    "paper title. Do not claim anything the papers do not state. 2-3 sentences, concrete."
)

PROMPT = """STUDENT interests:
{student}

PROFESSOR: {professor}
Matching papers (the only evidence you may use):
{papers}

Write 2-3 sentences on why this professor fits the student, citing at least one paper title."""


def _format_papers(hit_papers):
    lines = []
    for p in hit_papers:
        title = str(p.get("Paper Title", "")).strip()
        abstract = str(p.get("Abstract", "")).strip()[:400]
        lines.append(f"- \"{title}\": {abstract}")
    return "\n".join(lines)


def _call_llm(system, user, model=MODEL, retries=3):
    from openai import OpenAI
    client = OpenAI()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=160,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def explain(student_text, professor_name, hit_papers, model=MODEL):
    """hit_papers: list of dict-like rows with 'Paper Title' and 'Abstract'."""
    user = PROMPT.format(student=student_text[:1200],
                         professor=professor_name,
                         papers=_format_papers(hit_papers))
    return _call_llm(SYSTEM, user, model)


if __name__ == "__main__":
    demo = [{"Paper Title": "Data-driven constitutive laws for solids",
             "Abstract": "We learn material behavior from data using neural networks..."}]
    print(_format_papers(demo))
    print("format OK — set OPENAI_API_KEY and call explain() to run for real")