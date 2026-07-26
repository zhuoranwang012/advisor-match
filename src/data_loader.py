"""[WRITE-NEW] Load CSVs and parse the stringified-list columns.

Student list-fields (seed_keywords/research_interests/methods_tools/profile_keywords)
are stored as "['a','b']" strings — parse with ast.literal_eval into real lists.
"""
import ast
import pandas as pd

def parse_list(x):
    if pd.isna(x):
        return []
    try:
        return [str(t).strip() for t in ast.literal_eval(x)]
    except Exception:
        return [t.strip(" '\"") for t in str(x).strip("[]").split(",") if t.strip()]

# TODO: load_students(), load_professors(), load_papers()
