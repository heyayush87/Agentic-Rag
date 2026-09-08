"""Streamlit demo UI — the thing to screen-share in an interview.

    streamlit run src/retailiq/ui/streamlit_app.py

Surfaces the agent's decision trace next to the answer, which is what makes
the "agentic" behaviour visible rather than a black box.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit executes this file as a script, not as a package module, so the
# source root has to be importable before `retailiq` resolves. Harmless when
# the package is already pip-installed.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(_SRC))

from retailiq import __version__  # noqa: E402
from retailiq.core.exceptions import RetailIQError, VectorStoreNotFoundError  # noqa: E402
from retailiq.services.rag_service import RAGService  # noqa: E402

st.set_page_config(page_title="RetailIQ", page_icon="🛒", layout="centered")


@st.cache_resource(show_spinner=False)
def _service() -> RAGService:
    """One service per session — the graph and embedding model are expensive."""
    return RAGService()


st.title("🛒 RetailIQ")
st.caption(
    "A self-correcting agentic RAG assistant for retail operations. It routes each "
    "question, grades its own retrieved documents, rewrites weak queries, and checks "
    "the answer for hallucination before responding."
)

with st.sidebar:
    st.header("About")
    st.write(
        "Ask about returns, Clubcard/loyalty, store operations, supplier and "
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

    st.subheader("Index")
    if st.button("Build / rebuild knowledge index"):
        from retailiq.ingestion.pipeline import build_index

        with st.spinner("Ingesting documents…"):
            try:
                result = build_index(reset=True)
            except RetailIQError as exc:
                st.error(exc.message)
            else:
                st.cache_resource.clear()
                st.success(result.describe())

    st.divider()
    st.caption(f"RetailIQ v{__version__}")

question = st.text_input(
    "Ask a question", placeholder="e.g. Can I return opened washing-up liquid?"
)

if st.button("Ask", type="primary") and question:
    try:
        with st.spinner("The agent is reasoning…"):
            result = _service().answer(question)
    except VectorStoreNotFoundError:
        st.warning("No knowledge index yet. Build one from the sidebar first.")
    except RetailIQError as exc:
        st.error(f"{exc.code}: {exc.message}")
    else:
        st.subheader("Answer")
        st.write(result.answer)

        left, middle, right = st.columns(3)
        left.metric("Route", str(result.route))
        middle.metric("Grounded", "yes" if result.grounded else "no")
        right.metric("Latency", f"{result.latency_ms} ms")

        if result.sources:
            st.caption("Sources: " + ", ".join(result.sources))

        with st.expander("🔍 Agent decision trace", expanded=True):
            for step in result.trace:
                st.markdown(f"- {step.detail}")

        with st.expander("📄 Retrieved context"):
            st.text(result.context)
