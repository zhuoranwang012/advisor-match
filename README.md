# Research-Interest Semantic Matching Engine

Students describe their interests in plain language — *"machine learning for materials"* —
while professors publish in specialized terms — *"data-driven constitutive laws for solids."*
Same idea, **zero shared vocabulary.**

Across 4,000 student profiles, **96% share less than 0.1 keyword overlap with their
best-matching professor.** This vocabulary gap is exactly why surface keyword search fails
here — and why this project uses **paper-level semantic retrieval** to bridge it, recommending
real faculty advisors with explanations grounded in their specific papers.

> **Data honesty:** Professors and papers are **real** (scraped from Semantic Scholar), so the
> recommendations are real. Student profiles are **LLM-synthesized** and used *only* as query
> inputs and a controllable evaluation set. The system makes **no claims about real students**;
> every recommended entity is a real professor.

## What it does
Student interest (free text) → retrieve most relevant **papers** → aggregate scores to their
**authors (professors)** → Top-5 advisor recommendations, each with a RAG-generated reason that
cites the specific matching papers.

## Method
- **Retrieval unit: papers, not professors.** Captures "one on-target paper" that a
  professor-level summary would dilute.
- **Encoder:** MiniLM (primary); RoBERTa comparison included.
- **Baselines:** random / keyword-overlap / semantic — to show semantic wins where keywords can't.

## Evaluation (independent of the embeddings, to avoid circularity)
- **Primary:** LLM-as-judge (GPT) on a sampled set of students.
- **Secondary:** research-area co-membership as a cheap full-set sanity check.
- **Metrics:** Recall@5 · NDCG@5 · MRR.
- **Result:** _TBD — fill in after local embedding run._

## Repo layout
| Path | Role |
|------|------|
| `data/raw/` | real CSVs (profs, papers) + synthetic student CSV |
| `src/` | data loading, embedding, retrieval, baselines, explanation, judge |
| `eval/` | proxy labels, metrics, eval runner |
| `notebooks/` | EDA + interactive topic map |
| `app/` | optional Streamlit demo |
| `legacy/` | archived original RA scripts (provenance) |

## Quickstart
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...   # for RAG explanation + LLM judge
# 1. embed papers + students   (needs the model; run locally)
python src/embed.py
# 2. run retrieval + baselines
python src/retrieve.py
# 3. evaluate
python eval/run_eval.py
```
