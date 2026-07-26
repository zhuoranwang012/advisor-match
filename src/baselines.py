"""Non-semantic baselines: random and keyword-overlap.

These need no embedding model, so they run anywhere. The keyword baseline is the
'foil' — surface Jaccard overlap, which the vocabulary gap makes weak on purpose.
"""
import numpy as np


def _words(list_of_phrases):
    w = set()
    for kw in list_of_phrases:
        for t in str(kw).replace("-", " ").lower().split():
            if len(t) > 2:
                w.add(t)
    return w


def student_words(row):
    return _words(row["research_interests"]) | _words(row["profile_keywords"])


def prof_words(row):
    return _words(row["top_keywords"])


def random_recommend(prof_names, k=5, rng=None):
    rng = rng or np.random.default_rng(0)
    return list(rng.choice(prof_names, size=k, replace=False))


def keyword_recommend(s_words, profs_df, pwords, k=5):
    """Jaccard word overlap between a student and every professor -> Top-K names."""
    scores = np.array([
        len(s_words & pw) / (len(s_words | pw) or 1) for pw in pwords
    ])
    idx = np.argsort(scores)[-k:][::-1]
    return [profs_df.iloc[j]["professor_name"] for j in idx]
