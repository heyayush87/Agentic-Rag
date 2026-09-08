"""Wires the nodes into a LangGraph state machine — the agentic RAG controller.

Flow:

    START
      │
      ▼
    route_question ──"answer"──────────────► direct_answer ─► END
      │ "web_search"                              ▲
      ├───────────────► do_web_search ─► generate │
      │ "vectorstore"                        │
      ▼                                      │
    retrieve ─► grade_documents ─┬─(no relevant docs & rewrites left)─► rewrite ─► retrieve
                                 └─(have relevant docs OR out of rewrites)─► generate
    generate ─► grade_generation ─┬─ "useful" ─────────► END
                                  ├─ "not_grounded" ──► generate (regenerate)
                                  └─ "not_useful" ────► rewrite ─► retrieve (if budget left) else END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

import config
from src import nodes
from src.nodes import GraphState


# --- conditional edge functions ----------------------------------------
def _after_route(state: GraphState) -> str:
    return state["route"]


def _after_grade_docs(state: GraphState) -> str:
    has_docs = len(state.get("documents", [])) > 0
    if has_docs:
        return "generate"
    if state.get("rewrites", 0) < config.MAX_QUERY_REWRITES:
        return "rewrite"
    # Out of rewrites and still nothing relevant → answer honestly from empty context.
    return "generate"


def _after_generation(state: GraphState) -> str:
    verdict = nodes.grade_generation(state)
    if verdict == "useful":
        return "done"
    if verdict == "not_grounded":
        return "regenerate"
    # not_useful → try a rewrite if we still have budget
    if state.get("rewrites", 0) < config.MAX_QUERY_REWRITES:
        return "rewrite"
    return "done"


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("route_question", nodes.route_question)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("rewrite", nodes.rewrite_query)
    g.add_node("web_search", nodes.do_web_search)
    g.add_node("generate", nodes.generate)
    g.add_node("direct_answer", nodes.direct_answer)

    g.add_edge(START, "route_question")
    g.add_conditional_edges(
        "route_question",
        _after_route,
        {"vectorstore": "retrieve", "web_search": "web_search", "answer": "direct_answer"},
    )

    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        _after_grade_docs,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("web_search", "generate")

    g.add_conditional_edges(
        "generate",
        _after_generation,
        {"done": END, "regenerate": "generate", "rewrite": "rewrite"},
    )
    g.add_edge("direct_answer", END)

    return g.compile()


# Build lazily so importing this module has no side effects.
_APP = None


def get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def answer_question(question: str) -> dict:
    """Convenience entry point. Returns the full final state (answer + trace)."""
    app = get_app()
    init: GraphState = {
        "question": question,
        "original_question": question,
        "rewrites": 0,
        "trace": [],
    }
    final = app.invoke(init, config={"recursion_limit": 25})
    return final
