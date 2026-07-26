"""Load CSVs and parse the stringified-list columns.

Student list-fields are stored as "['a','b']" strings — parse into real lists.
Paths are relative to the repo root.
"""
import ast
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

STUDENTS_CSV = RAW / "students_virtual_engineering_profiles_v2_less_ai.csv"
PROFS_CSV    = RAW / "professors_profiles.csv"
PAPERS_CSV   = RAW / "papers_with_keywords_k10.csv"

LIST_COLS = ["seed_keywords", "research_interests", "methods_tools", "profile_keywords"]


def parse_list(x):
    """Turn "['a','b']" (or a plain string) into a list of strings."""
    if pd.isna(x):
        return []
    try:
        return [str(t).strip() for t in ast.literal_eval(x)]
    except Exception:
        return [t.strip(" '\"") for t in str(x).strip("[]").split(",") if t.strip()]


def load_students():
    df = pd.read_csv(STUDENTS_CSV)
    for c in LIST_COLS:
        if c in df.columns:
            df[c] = df[c].apply(parse_list)
    return df


def load_professors():
    df = pd.read_csv(PROFS_CSV)
    df["top_keywords"] = df["top_keywords"].apply(parse_list)
    df["prof_text"] = df["prof_text"].astype(str)
    return df


def load_papers():
    return pd.read_csv(PAPERS_CSV)


if __name__ == "__main__":
    s, p, pa = load_students(), load_professors(), load_papers()
    print(f"students {s.shape} | professors {p.shape} | papers {pa.shape}")
    print("sample student research_interests:", s.iloc[0]["research_interests"][:4])
