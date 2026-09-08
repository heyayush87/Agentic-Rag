"""The graph compiles and its wiring is intact — no credentials required."""

from __future__ import annotations

import pytest

from retailiq.agent.graph import build_graph, get_compiled_graph, reset_compiled_graph

pytestmark = pytest.mark.integration


def test_graph_compiles_without_any_provider_configured() -> None:
    """Construction must be side-effect free: no LLM calls, no vector store."""
    assert build_graph() is not None


def test_every_node_is_registered() -> None:
    nodes = set(build_graph().get_graph().nodes)
    for expected in (
        "contextualize",
        "route_question",
        "retrieve",
        "grade_documents",
        "rewrite",
        "web_search",
        "generate",
        "direct_answer",
    ):
        assert expected in nodes, f"missing node: {expected}"


def test_self_correction_loops_exist() -> None:
    """The corrective edges are what make this agentic rather than linear."""
    graph = build_graph().get_graph()
    edges = {(e.source, e.target) for e in graph.edges}

    assert ("contextualize", "route_question") in edges, (
        "follow-ups must be resolved before routing sees the question"
    )
    assert ("rewrite", "retrieve") in edges, "rewrite must feed back into retrieval"
    assert ("generate", "generate") in edges, "ungrounded answers must regenerate"
    assert ("generate", "rewrite") in edges, "unhelpful answers must re-query"


def test_compiled_graph_is_cached() -> None:
    reset_compiled_graph()
    assert get_compiled_graph() is get_compiled_graph()
    reset_compiled_graph()
