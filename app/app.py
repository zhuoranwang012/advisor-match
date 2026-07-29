"""Streamlit demo for the advisor-matching engine.

Run locally:
    streamlit run app/app.py

Core recommendation works with NO API key. RAG explanations are an optional add-on:
tick the box and paste an OpenAI key in the sidebar (kept in session only, never stored).
"""
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_students, load_professors, load_papers
from retrieve import build_student_query, paper_text, recommend_one
from embed import encode

st.set_page_config(page_title="Advisor Match", page_icon="🎓", layout="centered")

st.markdown("""
<style>
/* Larger sidebar type */
section[data-testid="stSidebar"] { font-size: 1.05rem; }
section[data-testid="stSidebar"] h2 { font-size: 1.6rem; font-weight: 700; }
section[data-testid="stSidebar"] label { font-size: 1.05rem !important; }

/* Clean, tight main title */
h1 { font-weight: 800; letter-spacing: -0.5px; }

/* Result cards: soft rounded corners + subtle shadow (playful but restrained) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.06);
    transition: box-shadow .2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 4px 18px rgba(37,99,235,0.14);
}

/* Professor name sizing */
.prof-name { font-size: 1.35rem; font-weight: 700; color: #0F172A; }
.prof-inst { font-size: 0.95rem; color: #64748B; font-style: italic; }

/* Top-1 crown highlight */
.top1-name { font-size: 1.5rem; font-weight: 800; color: #2563EB; }
.match-top1 { font-size: 2rem; font-weight: 800; color: #EF4444; text-align: right; }
.match-normal { font-size: 1.6rem; font-weight: 700; color: #0F172A; text-align: right; }
.match-label { font-size: 0.8rem; color: #94A3B8; text-align: right; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_students(), load_professors(), load_papers()


@st.cache_resource
def get_paper_embeddings(encoder="minilm"):
    """Load saved paper vectors if present, else compute once and cache."""
    path = ROOT / "outputs" / f"paper_embs_{encoder}.npy"
    if path.exists():
        return np.load(path)
    _, _, papers = get_data()
    with st.spinner("First run: encoding the paper corpus (one-time, ~1 min)…"):
        embs = encode(paper_text(papers), encoder)
    path.parent.mkdir(exist_ok=True)
    np.save(path, embs)
    return embs


# ---------- header ----------
st.title("🎓 Research-Interest Advisor Match")
st.caption(
    "Students speak in plain language; professors publish in specialized terms. "
    "This tool bridges that vocabulary gap with paper-level semantic retrieval — "
    "recommending **real** faculty, with matches grounded in their **real** papers."
)

students, profs, papers = get_data()

# ---------- sidebar ----------
with st.sidebar:
    st.header("Query")
    mode = st.radio("Input mode", ["Free text", "Sample student"])
    if mode == "Free text":
        query_text = st.text_area("Describe research interests",
                                  value="soft robotics and control for prosthetics",
                                  height=100)
        who = "(free text)"
    else:
        sid = st.selectbox("Pick a synthetic student", students["student_id"].tolist())
        row = students[students["student_id"] == sid].iloc[0]
        query_text = build_student_query(row)
        who = f'{sid} · {row["major"]}'
        st.caption(f"Query built from profile:\n\n{query_text[:220]}…")

    k = st.slider("How many advisors", 3, 10, 5)

    st.divider()
    want_explanations = st.checkbox("Generate RAG explanations", value=False)
    api_key_input = ""
    if want_explanations:
        st.caption("Grounded reason per match (RAG). Needs an OpenAI key — "
                   "kept in session only, never stored.")
        api_key_input = st.text_input("OpenAI API key", type="password", placeholder="sk-…")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
        else:
            st.caption("↑ enter a key to enable explanations")

    go = st.button("Recommend", type="primary")

st.divider()

# ---------- results ----------
if go:
    with st.spinner("Encoding query and retrieving papers…"):
        paper_embs = get_paper_embeddings("minilm")
        q_emb = encode([query_text], "minilm")[0]
        top = recommend_one(q_emb, paper_embs, papers, k)

    st.subheader(f"Top {k} advisors for {who}")

    explain_fn = None
    if want_explanations and api_key_input:
        try:
            from explain import explain as explain_fn
        except Exception as e:
            st.warning(f"Explanation module unavailable: {e}")

    for rank, (name, score, paper) in enumerate(top, 1):
        inst = paper.get("Institution", "")
        title = str(paper.get("Paper Title", ""))
        year = paper.get("Year", "")
        cites = paper.get("Citation Count", "")
        link = paper.get("Final Link") or paper.get("Semantic Scholar URL") or ""

        with st.container(border=True):
            is_top1 = (rank == 1)
            c1, c2 = st.columns([4, 1])
            crown = "👑 " if is_top1 else ""
            name_cls = "top1-name" if is_top1 else "prof-name"
            c1.markdown(
                f'<span class="{name_cls}">{crown}{rank}. {name}</span><br>'
                f'<span class="prof-inst">{inst}</span>',
                unsafe_allow_html=True,
            )
            score_cls = "match-top1" if is_top1 else "match-normal"
            c2.markdown(
                f'<div class="match-label">match</div>'
                f'<div class="{score_cls}">{score:.2f}</div>',
                unsafe_allow_html=True,
            )
            paper_line = f"📄 [{title}]({link})" if link else f"📄 *{title}*"
            meta = " · ".join(str(x) for x in [year, f"{cites} citations"] if str(x).strip())
            st.markdown(paper_line + (f"  \n<sub>{meta}</sub>" if meta else ""),
                        unsafe_allow_html=True)
            if explain_fn:
                with st.spinner("Explaining…"):
                    try:
                        st.info(explain_fn(query_text, name, [paper]))
                    except Exception as e:
                        st.caption(f"(explanation unavailable: {e})")

    st.caption("Synthetic students are queries only; recommended professors and cited papers are real.")
else:
    st.info("Set a query in the sidebar and click **Recommend**. "
            "No API key needed for recommendations.")