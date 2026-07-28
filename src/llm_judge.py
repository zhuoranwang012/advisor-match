"""LLM-as-judge (GPT), provider-agnostic interface.

judge_pair(student_text, professor_text) -> relevance score 0/1/2
  0 = not a fit, 1 = partial/adjacent fit, 2 = strong fit

Design notes
------------
- Reads OPENAI_API_KEY from the environment; never hardcode a key.
- Kept SEPARATE from explain.py so the judge never grades its own generated text.
- `_call_llm` is the only provider-specific function — swap it to use a different
  vendor without touching the scoring logic (this is the "provider-agnostic" part).
- Cheap model by default; a judge only reads two short texts and returns one number.

RUN LOCALLY (needs network + API key). Sandbox-checked for syntax only.
"""
import json
import os
import re
import time

MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")   # cheap tier is plenty for judging

SYSTEM = (
    "You evaluate whether a professor is a good research-advising match for a student. "
    "Score STRICTLY on research-topic fit. Return only an integer: "
    "0 = not a fit, 1 = partial or adjacent fit, 2 = strong fit."
)

PROMPT = """STUDENT (interests):
{student}

PROFESSOR (research):
{professor}

How well does this professor's research match the student's interests?
Answer with a single integer 0, 1, or 2. No other text."""


def _call_llm(system, user, model=MODEL, retries=3):
    """The ONLY provider-specific code. Returns the model's raw text reply."""
    from openai import OpenAI
    client = OpenAI()   # reads OPENAI_API_KEY from env
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0,
                max_tokens=5,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _parse_score(text):
    m = re.search(r"[012]", text or "")
    return int(m.group()) if m else 0


def judge_pair(student_text, professor_text, model=MODEL):
    """Return an integer relevance score 0/1/2 for one (student, professor) pair."""
    reply = _call_llm(SYSTEM, PROMPT.format(student=student_text[:1500],
                                            professor=professor_text[:1500]), model)
    return _parse_score(reply)


def judge_rankings(students_df, profs_df, rankings, sample_ids, k=5,
                   relevant_threshold=1, cache_path=None):
    """Judge the Top-k of `rankings` for each student in sample_ids.

    Returns {student_id: {professor_name: score}}.
    A professor counts as 'relevant' if score >= relevant_threshold (used by metrics).
    Caches to disk so a crash/rate-limit doesn't lose progress.
    """
    sview = students_df.set_index("student_id")
    pview = profs_df.set_index("professor_name")

    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path))

    for n, sid in enumerate(sample_ids, 1):
        if sid in cache:
            continue
        s_text = str(sview.loc[sid].get("interest_profile", ""))
        scores = {}
        for name in rankings.get(sid, [])[:k]:
            if name in pview.index:
                scores[name] = judge_pair(s_text, str(pview.loc[name].get("prof_text", "")))
        cache[sid] = scores
        if cache_path and n % 10 == 0:
            json.dump(cache, open(cache_path, "w"))
            print(f"  judged {n}/{len(sample_ids)}")

    if cache_path:
        json.dump(cache, open(cache_path, "w"))
    return cache


def to_relevance(judged, threshold=1):
    """Convert judged scores -> {student_id: set(relevant professor names)} for metrics."""
    return {sid: {n for n, sc in d.items() if sc >= threshold}
            for sid, d in judged.items()}


if __name__ == "__main__":
    # tiny smoke test of parsing (no API call)
    assert _parse_score("2") == 2 and _parse_score("score: 1") == 1 and _parse_score("x") == 0
    print("parse OK — set OPENAI_API_KEY and call judge_rankings() to run for real")