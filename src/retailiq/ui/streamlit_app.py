"""RetailIQ chat interface.

    streamlit run src/retailiq/ui/streamlit_app.py

A conversational UI in the shape people expect from ChatGPT or Claude:
persistent chat threads in a sidebar, message bubbles, and multi-turn memory
so follow-ups like "what about food?" resolve against what was already asked.

Everything shown is read from the running system — indexed document names,
chunk counts, provider, starter questions — so nothing on screen can drift
out of step with what the app is actually doing.
"""

from __future__ import annotations

import json
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
from retailiq.core.settings import get_settings  # noqa: E402
from retailiq.domain.models import ChatTurn, Conversation  # noqa: E402
from retailiq.services.conversation_store import ConversationStore  # noqa: E402
from retailiq.services.rag_service import RAGService  # noqa: E402

st.set_page_config(page_title="RetailIQ", page_icon="🛒", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources — expensive to build, safe to share across reruns
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _service() -> RAGService:
    return RAGService()


@st.cache_resource(show_spinner=False)
def _store() -> ConversationStore:
    return ConversationStore()


@st.cache_data(ttl=60, show_spinner=False)
def _index_stats() -> dict:
    """Live index facts. Short TTL so a re-ingest shows up without a restart."""
    return _service().index_stats()


@st.cache_data(show_spinner=False)
def _starter_questions() -> list[str]:
    """Starter prompts taken from the project's own evaluation dataset.

    Read from `data/eval/eval_questions.json` rather than hard-coded, so every
    suggestion is a question the knowledge base is known to answer. Hard-coded
    examples rot the moment the documents change.
    """
    path = get_settings().paths.eval_dataset
    try:
        cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [c["question"] for c in cases if isinstance(c, dict) and "question" in c]


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
def _current() -> Conversation:
    """The open conversation, creating one on first load."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = Conversation()
    return st.session_state.conversation


def _open(conversation: Conversation) -> None:
    st.session_state.conversation = conversation
    st.rerun()


def _new_chat() -> None:
    st.session_state.conversation = Conversation()
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    store = _store()
    active = _current()

    with st.sidebar:
        st.markdown("### 🛒 RetailIQ")

        if st.button("＋  New chat", use_container_width=True, type="primary"):
            _new_chat()

        st.markdown("#### Recent")
        recent = store.list_recent(limit=30)

        if not recent:
            st.caption("No conversations yet. Ask something to start one.")

        for conversation in recent:
            is_active = conversation.id == active.id
            row, remove = st.columns([0.82, 0.18])

            with row:
                if st.button(
                    ("▸ " if is_active else "") + conversation.title,
                    key=f"open_{conversation.id}",
                    use_container_width=True,
                    disabled=is_active,
                    help=conversation.updated_at.strftime("%d %b %Y, %H:%M UTC"),
                ):
                    _open(conversation)

            with remove:
                if st.button("🗑", key=f"del_{conversation.id}", help="Delete"):
                    store.delete(conversation.id)
                    if is_active:
                        st.session_state.conversation = Conversation()
                    st.rerun()

        st.divider()

        # --- live system state, not static text --------------------------
        stats = _index_stats()
        settings = get_settings()

        with st.expander("Knowledge base", expanded=False):
            if stats["built"]:
                left, right = st.columns(2)
                left.metric("Documents", stats["documents"])
                right.metric("Chunks", stats["chunks"])
                for name in stats["sources"]:
                    st.caption(f"• {name}")
            else:
                st.warning("No index built yet.")

            if st.button("Rebuild index", use_container_width=True):
                from retailiq.ingestion.pipeline import build_index

                with st.spinner("Ingesting documents…"):
                    try:
                        result = build_index(reset=True)
                    except RetailIQError as exc:
                        st.error(exc.message)
                    else:
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.success(result.describe())

        with st.expander("Configuration", expanded=False):
            st.caption(f"Chat · `{settings.llm.provider}`")
            st.caption(f"Embeddings · `{settings.embeddings.provider}`")
            st.caption(f"Search · `{settings.tools.web_search_provider}`")
            st.caption(f"Top-k · `{settings.retrieval.top_k}`")

        st.caption(f"v{__version__}")


# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------
def _render_turn(turn: ChatTurn) -> None:
    with st.chat_message(turn.role, avatar="🧑" if turn.role == "user" else "🛒"):
        st.markdown(turn.content)

        if turn.role != "assistant":
            return

        if turn.sources:
            st.caption("Sources: " + ", ".join(f"`{s}`" for s in turn.sources))

        if turn.trace:
            label = f"Agent reasoning · {len(turn.trace)} steps"
            if turn.latency_ms:
                label += f" · {turn.latency_ms / 1000:.1f}s"
            with st.expander(label):
                for index, step in enumerate(turn.trace, start=1):
                    st.markdown(f"`{index}` {step}")
                meta = []
                if turn.route:
                    meta.append(f"route `{turn.route}`")
                meta.append("grounded ✅" if turn.grounded else "grounded ⚠️")
                st.caption(" · ".join(meta))


def _render_welcome() -> None:
    """Empty state: a heading plus real, answerable starter questions."""
    st.markdown("## What would you like to know?")
    st.caption(
        "Ask about returns, loyalty, store operations, supplier logistics, "
        "or the product catalogue."
    )

    questions = _starter_questions()
    if not questions:
        return

    st.write("")
    columns = st.columns(2)
    for index, question in enumerate(questions[:4]):
        with columns[index % 2]:
            if st.button(question, key=f"starter_{index}", use_container_width=True):
                st.session_state.pending = question
                st.rerun()


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------
def _answer(question: str) -> None:
    """Run the agent for `question` and persist both turns."""
    conversation = _current()
    store = _store()

    conversation.add(ChatTurn(role="user", content=question))
    _render_turn(conversation.turns[-1])

    with st.chat_message("assistant", avatar="🛒"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")

        try:
            # Everything before the message just added is the memory the agent
            # uses to resolve references in this one.
            result = _service().answer(question, history=conversation.history_before_last())
        except VectorStoreNotFoundError:
            placeholder.warning(
                "No knowledge index yet — build one from **Knowledge base** in the sidebar."
            )
            conversation.turns.pop()  # don't persist a question we never answered
            return
        except RetailIQError as exc:
            placeholder.error(f"{exc.code}: {exc.message}")
            conversation.turns.pop()
            return

        placeholder.empty()

    conversation.add(
        ChatTurn(
            role="assistant",
            content=result.answer,
            sources=result.sources,
            trace=[step.detail for step in result.trace],
            route=result.route,
            grounded=result.grounded,
            latency_ms=result.latency_ms,
        )
    )
    store.save(conversation)
    st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
_render_sidebar()

conversation = _current()

if conversation.is_empty:
    _render_welcome()
else:
    for turn in conversation.turns:
        _render_turn(turn)

typed = st.chat_input("Ask about returns, loyalty, store operations…")

# A starter button sets `pending`; the chat box sets `typed`. Both land here.
pending = st.session_state.pop("pending", None)
question = typed or pending

if question:
    _answer(question)
