"""Research-area definitions and mappings (for the secondary evaluation label).

- CLUSTER_TO_FACTORY: paper cluster_id -> one of 6 research areas.
- professor_area(): each professor's DOMINANT area, from their papers' clusters.
- MAJOR_TO_AREAS: student major -> set of plausible areas (a coarse, transparent proxy).
  A student may map to several areas; relevance = professor's area is in that set.
"""
from collections import Counter
import pandas as pd

CLUSTER_TO_FACTORY = {
    0: "Materials & Solid Mechanics",
    3: "Thermal-Fluid & Transport Phenomena",
    5: "Advanced Manufacturing & Devices",
    1: "Robotics & Intelligent Systems",
    2: "Robotics & Intelligent Systems",
    9: "Robotics & Intelligent Systems",
    4: "Biomedical & Biomechanical Engineering",
    8: "Biomedical & Biomechanical Engineering",
    6: "Systems, Education & Interdisciplinary",
    7: "Systems, Education & Interdisciplinary",
}

# Student major -> plausible research areas. Coarse proxy, editable.
MAT = "Materials & Solid Mechanics"
THF = "Thermal-Fluid & Transport Phenomena"
MFG = "Advanced Manufacturing & Devices"
ROB = "Robotics & Intelligent Systems"
BIO = "Biomedical & Biomechanical Engineering"
SYS = "Systems, Education & Interdisciplinary"

MAJOR_TO_AREAS = {
    "Computer Science": {ROB},
    "Computer Engineering": {ROB, MFG},
    "Electrical Engineering": {ROB, MFG},
    "Industrial Engineering": {SYS},
    "Systems Engineering & Design": {SYS},
    "Manufacturing and Design Engineering": {MFG},
    "Biomedical Engineering": {BIO},
    "Bioengineering": {BIO},
    "Neural Engineering": {BIO, ROB},
    "Agricultural & Biological Engr": {BIO},
    "Civil Engineering": {MAT},
    "Environmental Engineering": {THF},
    "Mechanical Engineering": {MAT, THF, MFG},
    "Aerospace Engineering": {THF, MAT},
    "Chemical Engineering": {THF},
    "Nuclear, Plasma, Radiological Engr": {THF, MAT},
    "Engineering Mechanics": {MAT},
    "Materials Science and Engineering": {MAT},
    "Materials Science & Engr": {MAT},
    "Physics": {MAT},
    "Engineering Physics": {MAT},
    "Applied Math": {SYS},
    "Computer Science & Bioengineering": {ROB, BIO},
    "Computer Science & Physics": {ROB, MAT},
    "McCormick Integrated Engineering Studies": {SYS},
    # Undeclared / Undecided -> no honest area; excluded from area eval (None).
    "Engineering Undeclared": None,
    "Undecided": None,
}


def professor_area(papers_df):
    """Return {professor_name: dominant_area} from their papers' clusters."""
    df = papers_df.copy()
    df["area"] = df["cluster_id"].map(CLUSTER_TO_FACTORY)
    out = {}
    for name, g in df.groupby("Professor Name (Original)"):
        areas = g["area"].dropna()
        if len(areas):
            out[name] = Counter(areas).most_common(1)[0][0]
    return out


def student_areas(major):
    """Return the area set for a major, or None if it can't be assigned."""
    return MAJOR_TO_AREAS.get(major, None)


if __name__ == "__main__":
    from data_loader import load_papers
    pa = load_papers()
    parea = professor_area(pa)
    print("professors with an area:", len(parea))
    print("area distribution:", Counter(parea.values()))
