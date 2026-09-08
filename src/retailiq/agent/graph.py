"""Graph assembly — wiring nodes and edges into the agent controller.

START
  │
  ▼
route_question ──"answer"─────────────────► direct_answer ──► END
  │ "web_search"
  ├──────────────► web_search ──► generate
  │ "vectorstore"
  ▼
retrieve ──► grade_documents ─┬─ nothing relevant & budget left ─► rewrite ─┐
                              └─ relevant, or out of budget ──────► generate│
                                                               ▲            │
                                                               └────────────┘
generate ──► grade_generation ─┬─ "useful" ──────────────────────► END
                               ├─ "not_grounded" ──► generate (retry)
                               └─ "not_useful" ────► rewrite ──► retrieve
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from retailiq.agent import edges, nodes
from retailiq.agent.state import AgentState
from retailiq.core.logging import get_logger

logger = get_logger(__name__)

# Node identifiers. Constants because LangGraph resolves these as strings at
# runtime — a typo in a routing map would otherwise surface as a confusing
# KeyError deep inside an invocation rather than at build time.
CONTEXTUALIZE = "contextualize"
ROUTE_QUESTION = "route_question"
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
REWRITE = "rewrite"
WEB_SEARCH = "web_search"
GENERATE = "generate"
DIRECT_ANSWER = "direct_answer"


def build_graph() -> Any:
    """Construct and compile the agent state machine.

    Pure and side-effect free: it makes no LLM calls and touches no vector
    store, so tests can assert the topology compiles without any credentials.
    """
    graph = StateGraph(AgentState)

    graph.add_node(CONTEXTUALIZE, nodes.contextualize_question)
    graph.add_node(ROUTE_QUESTION, nodes.route_question)
    graph.add_node(RETRIEVE, nodes.retrieve)
    graph.add_node(GRADE_DOCUMENTS, nodes.grade_documents)
    graph.add_node(REWRITE, nodes.rewrite_query)
    graph.add_node(WEB_SEARCH, nodes.web_search)
    graph.add_node(GENERATE, nodes.generate)
    graph.add_node(DIRECT_ANSWER, nodes.direct_answer)

    # Resolve conversational references before anything else looks at the
    # question, so routing and retrieval both see a standalone form.
    graph.add_edge(START, CONTEXTUALIZE)
    graph.add_edge(CONTEXTUALIZE, ROUTE_QUESTION)

    graph.add_conditional_edges(
        ROUTE_QUESTION,
        edges.route_decision,
        {
            edges.RETRIEVE: RETRIEVE,
            edges.WEB_SEARCH: WEB_SEARCH,
            edges.DIRECT_ANSWER: DIRECT_ANSWER,
        },
    )

    graph.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    graph.add_conditional_edges(
        GRADE_DOCUMENTS,
        edges.documents_decision,
        {edges.GENERATE: GENERATE, edges.REWRITE: REWRITE},
    )

    # The corrective loop: a rewritten query goes straight back to retrieval.
    graph.add_edge(REWRITE, RETRIEVE)
    graph.add_edge(WEB_SEARCH, GENERATE)

    graph.add_conditional_edges(
        GENERATE,
        edges.generation_decision,
        {edges.DONE: END, edges.REGENERATE: GENERATE, edges.REWRITE: REWRITE},
    )
    graph.add_edge(DIRECT_ANSWER, END)

    return graph.compile()


# Compiling is cheap but not free, and the graph is stateless once built, so
# one instance is shared per process. Built lazily so that importing this
# module has no side effects.
_compiled: Any | None = None


def get_compiled_graph() -> Any:
    """Return the process-wide compiled graph, building it on first use."""
    global _compiled
    if _compiled is None:
        logger.info("Compiling agent graph")
        _compiled = build_graph()
    return _compiled


def reset_compiled_graph() -> None:
    """Discard the cached graph. Used by tests that patch node behaviour."""
    global _compiled
    _compiled = None
