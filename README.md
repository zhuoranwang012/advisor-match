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
> every recommended entity is a real professor. This is, in effect, a **cold-start methodology**:
> how to build and evaluate a semantic matcher when no real user data exists yet.

## What it does
Student interest (free text) → retrieve most relevant **papers** → aggregate scores to their
**authors (professors)** → Top-5 advisor recommendations, each with a reason grounded in the
specific matching papers.

## Example

Query: *"soft robotics and control for prosthetics"*

| score | professor | grounded reason (cites a real paper) |
|---|---|---|
| 0.674 | **Ryan L. Truby** | Works on entirely soft, autonomous robots — *"An integrated design and fabrication strategy for entirely soft, autonomous robots"* — directly matching the student's soft-robotics interest. |
| 0.604 | **Todd D. Murphey** | *"Materializing Autonomy in Soft Robots across Scales"* — integrates computational intelligence with physical design, relevant to control for prosthetics. |
| 0.592 | **Mattia Gazzola** | *"Elastica: A Compliant Mechanics Environment for Soft Robotic Control"* — models continuum mechanics for controlling soft robots. |
| 0.542 | **Justin Yim** | *"Electrostatic footpads enable agile insect-scale soft robots"* — agile soft-robot locomotion. |

All four are genuine soft-robotics faculty, retrieved from a free-text query that shares
little surface vocabulary with the papers — the vocabulary gap in action.



## Method
- **Retrieval unit: papers, not professors.** Captures a single on-target paper that a
  professor-level summary would dilute; also yields the exact papers behind each match.
- **Encoder:** MiniLM (primary) vs. sentence-RoBERTa — MiniLM matches quality at ~17× faster
  encoding (11s vs. 3m11s for the paper corpus), so it is the practical choice.
- **Baselines:** random / keyword-overlap / semantic.

## Evaluation
Two layers, both **independent of the embeddings** (to avoid circularity):

**Primary — LLM-as-judge** (GPT scores each recommendation's fit, sampled students):

| method | Hit@5 | Prec@5 | NDCG@5 | MRR |
|---|---|---|---|---|
| random | 0.683 | 0.400 | 0.420 | 0.530 |
| keyword | 0.817 | 0.567 | 0.621 | 0.695 |
| **semantic** | **0.883** | **0.583** | **0.663** | **0.728** |

*Semantic wins on every metric.* (n = 60, relevance ≥ 1.)

**Secondary — research-area co-membership** (coarse proxy, full set). Here semantic's Hit@5
(0.66) is *lower* than keyword (0.78) — a deliberate finding: the coarse label **cannot see
cross-area precision matches** (e.g. an ML-for-materials professor for a CS student), so it
systematically under-credits semantic retrieval. Under the LLM judge that reads meaning, the
Hit@5 ranking **reverses** (semantic highest at 0.88) — confirming the proxy's blind spot
rather than a real weakness. This is why the LLM judge is the primary metric.

## Repo layout
| Path | Role |
|------|------|
| `data/raw/` | real CSVs (profs, papers) + synthetic student CSV |
| `src/` | data loading, embedding, retrieval, baselines, explanation, judge |
| `eval/` | proxy labels, metrics, eval runners (area + LLM judge) |
| `notebooks/` | EDA + interactive topic map |
| `app/` | optional Streamlit demo |
| `legacy/` | archived original RA scripts (provenance) |

## Quickstart
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...            # for the LLM judge (and RAG explanation)

python src/retrieve.py minilm                              # -> outputs/rankings_minilm.json
python eval/run_eval.py --rankings outputs/rankings_minilm.json          # area proxy eval
python eval/run_judge_eval.py --semantic outputs/rankings_minilm.json --sample 60   # LLM-judge eval
```