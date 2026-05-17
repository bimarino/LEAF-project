import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from search import search  


st.title("LEAF-BIANCA Semantic Search Demo")

query = st.text_input("Enter your query:", value="Explain research in simple words to my grandma")
k = st.slider("Number of results (k)", min_value=1, max_value=10, value=5)
use_reranker = st.checkbox("Use reranker", value=True)

if st.button("Search"):
    with st.spinner("Running search..."):
        results = search(query, top_k=k, use_reranker=use_reranker)

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if use_reranker:
        scores = results.get("scores", [[]])[0]
        score_label = "Score (reranker + difficulty bonus)"
    else:
        distances = results.get("distances", [[]])[0]
        scores = distances
        score_label = "Distance (lower is better)"

    for rank, (id_, doc, meta, score) in enumerate(zip(ids, docs, metas, scores), start=1):
        title = meta.get("title", "") if isinstance(meta, dict) else ""
        st.markdown(f"### Rank {rank} – {title}")
        st.write(f"ID: {id_}")
        st.write(f"{score_label}: {score}")
        st.write(doc[:300] + "…")
        st.markdown("---")