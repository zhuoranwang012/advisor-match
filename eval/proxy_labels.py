"""[WRITE-NEW] Secondary relevance label: research-area co-membership.

INDEPENDENT of embeddings AND of keyword overlap (avoids circularity).
professor area <- dominant paper cluster (src/areas.py)
student area   <- major-based mapping (coarse; documented as a proxy)
relevant(student, prof) = same area.
"""
# TODO: build_area_labels()
