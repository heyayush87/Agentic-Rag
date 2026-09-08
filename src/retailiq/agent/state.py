"""Shared state threaded through every node of the graph.

LangGraph merges each node's returned partial dict into this object. Keeping
it a `TypedDict` (rather than a pydantic model) is deliberate: LangGraph's
merge semantics operate on plain mappings, and the graph is hot-path code
where per-node revalidation would be wasted work. Validation happens at the
boundaries instead — settings on the way in, `QueryResult` on the way out.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document

from retailiq.core.enums import GenerationVerdict, Route


class AgentState(TypedDict, total=False):
    """Everything the agent knows while answering one question."""

    # --- inputs -----------------------------------------------------------
    question: str
    """Current query used for retrieval — mutated by the rewrite node."""

    original_question: str
    """What the user actually asked. Generation and relevance grading always
    judge against this, never the rewritten form, so a bad rewrite cannot
    silently change what gets answered."""

    history: list[tuple[str, str]]
    """Prior conversation as (role, content) pairs, oldest first.

    Retrieval is stateless — an embedding of "what about food?" matches
    nothing useful. So history is not passed to the retriever; it is used once,
    up front, to rewrite the follow-up into a standalone question.
    """

    contextualized: bool
    """Whether the opening node rewrote the question using history."""

    # --- routing ----------------------------------------------------------
    route: Route

    # --- retrieval --------------------------------------------------------
    documents: list[Document]
    context: str

    # --- generation -------------------------------------------------------
    generation: str
    verdict: GenerationVerdict

    # --- loop budgets -----------------------------------------------------
    rewrites: int
    """Query reformulations spent. Bounded by `MAX_QUERY_REWRITES`."""

    generation_attempts: int
    """Times the generate node has run. The first call is attempt 1; every
    further call is a retry after a failed grounding check, bounded by
    `MAX_GENERATION_RETRIES`. Without this counter the ungrounded → regenerate
    edge can loop forever when the context genuinely cannot support any
    answer."""

    # --- observability ----------------------------------------------------
    trace: list[str]


def initial_state(question: str, history: list[tuple[str, str]] | None = None) -> AgentState:
    """Build the entry state for a fresh question.

    Args:
        question: What the user just typed, verbatim.
        history: Prior (role, content) turns. Empty for the first message of a
            conversation, which lets the contextualisation node skip its LLM
            call entirely rather than paying for a no-op rewrite.
    """
    return AgentState(
        question=question,
        original_question=question,
        history=list(history or []),
        contextualized=False,
        documents=[],
        rewrites=0,
        generation_attempts=0,
        trace=[],
    )


def append_trace(state: AgentState, message: str) -> list[str]:
    """Return a new trace list with `message` appended.

    Returns a copy rather than mutating in place: LangGraph may retain or
    replay prior states, and shared mutable lists make that unsafe.
    """
    return [*state.get("trace", []), message]
