# Data

## Sources & honesty
- `professors_profiles.csv` — **real.** 101 engineering professors (UIUC + NU), aggregated
  from their papers.
- `papers_with_keywords_k10.csv` — **real.** ~1,992 papers scraped from Semantic Scholar
  (title, abstract, year, citations, cluster_id, keywords).
- `students_virtual_engineering_profiles_v2_less_ai.csv` — **LLM-synthesized.** 4,000 student
  profiles. Used ONLY as (a) query inputs and (b) a controllable evaluation set.
  **Not representative of any real student population.**

## Note on the vocabulary gap
Student profiles are written in colloquial / coursework terms; professor keywords are
specialized research phrases. Measured surface overlap is near-zero (96% of students < 0.1
keyword Jaccard with their best match). This motivates semantic retrieval and is why keyword
overlap is used as a *foil* baseline, not as a relevance label.
