"""Streamlit demo UI — the thing to screen-share in an interview.

    streamlit run app.py

Shows the answer AND the agent's decision trace (routing, grading, rewrites),
which is what makes the "agentic" behaviour visible rather than a black box.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Retail Agentic RAG", page_icon="🛒", layout="centered")

st.title("🛒 Retail Agentic RAG Assistant")
st.caption(
    "A self-correcting agentic RAG system for retail operations — it routes, "
    "grades its own retrieved documents, rewrites weak queries, and checks its "
    "answer for hallucination before responding."
)

with st.sidebar:
    st.header("About")
    st.write(
        "Ask about returns, Clubcard/loyalty, store operations, supplier & "
        "logistics procedures, or the product catalogue."
    )
    st.markdown(
        "**Example questions**\n"
        "- How long do I have to return an electrical item?\n"
        "- A chiller reads 7°C at opening — what do I do?\n"
        "- How many points do I need for a voucher?\n"
        "- What temperature must frozen deliveries arrive at?"
    )
    st.divider()
    st.write("First run? Build the index below.")
    if st.button("Build / rebuild knowledge index"):
        with st.spinner("Ingesting documents…"):
            from src.ingest import build_vectorstore
            build_vectorstore(reset=True)
        st.success("Index built.")


@st.cache_resource(show_spinner=False)
def _load_app():
    from src.graph import get_app
    return get_app()


question = st.text_input("Ask a question", placeholder="e.g. Can I return opened washing-up liquid?")

if st.button("Ask", type="primary") and question:
    from src.graph import answer_question

    with st.spinner("The agent is reasoning…"):
        result = answer_question(question)

    st.subheader("Answer")
    st.write(result.get("generation", "(no answer)"))

    with st.expander("🔍 Agent decision trace", expanded=True):
        for step in result.get("trace", []):
            st.markdown(f"- {step}")

    with st.expander("📄 Retrieved context"):
        st.text(result.get("context", "(none)"))
