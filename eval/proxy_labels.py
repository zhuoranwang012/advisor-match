"""Secondary relevance label: research-area co-membership.

Independent of both embeddings and keyword overlap (avoids circularity).
  professor area  <- dominant paper cluster        (src/areas.professor_area)
  student area(s) <- major-based mapping (coarse)   (src/areas.student_areas)
  relevant(student, prof) = prof's dominant area is in the student's area set.

Students whose major has no honest area (Undeclared/Undecided) are excluded.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from areas import professor_area, student_areas


def build_relevance(students_df, profs_df, papers_df):
    """Return {student_id: set(relevant_professor_names)} and the list of
    student_ids that are eligible for the area eval."""
    p_area = professor_area(papers_df)                 # name -> area
    prof_names = profs_df["professor_name"].tolist()

    relevance, eligible = {}, []
    for _, row in students_df.iterrows():
        areas = student_areas(row["major"])
        if not areas:                                  # None -> exclude
            continue
        rel = {n for n in prof_names if p_area.get(n) in areas}
        relevance[row["student_id"]] = rel
        eligible.append(row["student_id"])
    return relevance, eligible
