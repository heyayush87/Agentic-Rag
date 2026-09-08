"""The agent's control flow.

These are the highest-value tests in the suite: they pin the self-correction
behaviour and the loop budgets that stop it running away. All pure dict-in,
string-out — no LLM, no network.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from retailiq.agent import edges
from retailiq.agent.state import AgentState, append_trace, initial_state
from retailiq.core.enums import GenerationVerdict, Route
from retailiq.core.settings import Settings

pytestmark = pytest.mark.unit


@pytest.fixture
def settings() -> Settings:
    return Settings(agent={"max_query_rewrites": 2, "max_generation_retries": 2})  # type: ignore[arg-type]


# --- routing --------------------------------------------------------------
@pytest.mark.parametrize(
    ("route", "expected"),
    [
        (Route.VECTORSTORE, edges.RETRIEVE),
        (Route.WEB_SEARCH, edges.WEB_SEARCH),
        (Route.DIRECT_ANSWER, edges.DIRECT_ANSWER),
    ],
)
def test_route_decision(route: Route, expected: str) -> None:
    assert edges.route_decision(AgentState(route=route)) == expected


def test_route_defaults_to_retrieval_when_unset() -> None:
    """Missing route must not crash the graph."""
    assert edges.route_decision(AgentState()) == edges.RETRIEVE


# --- after document grading ----------------------------------------------
def test_relevant_documents_go_to_generation(settings: Settings) -> None:
    state = AgentState(documents=[Document(page_content="x")], rewrites=0)
    assert edges.documents_decision(state, settings) == edges.GENERATE


def test_no_documents_triggers_a_rewrite(settings: Settings) -> None:
    state = AgentState(documents=[], rewrites=0)
    assert edges.documents_decision(state, settings) == edges.REWRITE


def test_rewrite_budget_is_enforced(settings: Settings) -> None:
    """Out of budget with nothing relevant: answer honestly, don't loop."""
    state = AgentState(documents=[], rewrites=2)
    assert edges.documents_decision(state, settings) == edges.GENERATE


# --- after generation -----------------------------------------------------
def _patch_verdict(monkeypatch: pytest.MonkeyPatch, verdict: GenerationVerdict) -> None:
    monkeypatch.setattr(edges, "grade_generation", lambda _state: verdict)


def test_useful_answer_finishes(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    _patch_verdict(monkeypatch, GenerationVerdict.USEFUL)
    assert edges.generation_decision(AgentState(), settings) == edges.DONE


def test_ungrounded_answer_regenerates(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """A hallucination is retried against the same context."""
    _patch_verdict(monkeypatch, GenerationVerdict.NOT_GROUNDED)
    state = AgentState(generation_attempts=1)
    assert edges.generation_decision(state, settings) == edges.REGENERATE


def test_regeneration_budget_is_enforced(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """Without this cap an ungrounded answer loops forever burning tokens."""
    _patch_verdict(monkeypatch, GenerationVerdict.NOT_GROUNDED)
    state = AgentState(generation_attempts=3)  # 1 initial + 2 retries spent
    assert edges.generation_decision(state, settings) == edges.DONE


def test_useless_answer_rewrites_the_query(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """Grounded but unhelpful means the context was wrong: change the query."""
    _patch_verdict(monkeypatch, GenerationVerdict.NOT_USEFUL)
    state = AgentState(rewrites=0)
    assert edges.generation_decision(state, settings) == edges.REWRITE


def test_useless_answer_gives_up_when_out_of_rewrites(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    _patch_verdict(monkeypatch, GenerationVerdict.NOT_USEFUL)
    state = AgentState(rewrites=2)
    assert edges.generation_decision(state, settings) == edges.DONE


# --- state helpers --------------------------------------------------------
def test_initial_state_seeds_both_questions() -> None:
    state = initial_state("  Can I return this?  ")
    assert state["question"] == state["original_question"] == "  Can I return this?  "
    assert state["rewrites"] == 0
    assert state["generation_attempts"] == 0


def test_append_trace_does_not_mutate_the_original() -> None:
    """LangGraph may replay states; shared mutable lists would corrupt them."""
    state = AgentState(trace=["first"])
    updated = append_trace(state, "second")
    assert updated == ["first", "second"]
    assert state["trace"] == ["first"]
