"""Node behaviour, with a scripted fake chat model in place of a provider."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from retailiq.agent import nodes
from retailiq.agent.state import AgentState
from retailiq.core.enums import GenerationVerdict, Route

pytestmark = pytest.mark.unit


@pytest.fixture
def patch_model(monkeypatch: pytest.MonkeyPatch, fake_chat_model):
    """Install a scripted model in place of the real provider."""

    def _install(replies: list[str] | None = None, default: str = "yes"):
        model = fake_chat_model(replies=replies, default=default)
        monkeypatch.setattr(nodes, "get_chat_model", lambda *a, **k: model)
        return model

    return _install


# --- verdict parsing ------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("yes", "yes"),
        ("Yes.", "yes"),
        ('  "YES"  ', "yes"),
        ("no, the document is unrelated", "no"),
        ("**vectorstore**", "vectorstore"),
        ("", ""),
    ],
)
def test_first_token_survives_model_formatting(raw: str, expected: str) -> None:
    """Models add punctuation and markdown even when told not to."""
    assert nodes._first_token(raw) == expected


# --- conversational memory ------------------------------------------------
def test_first_message_skips_contextualisation(patch_model) -> None:
    """No history means no reference to resolve — don't pay for an LLM call."""
    model = patch_model(["should not be called"])
    result = nodes.contextualize_question(AgentState(question="hello", history=[]))

    assert result["contextualized"] is False
    assert model.calls == []


def test_follow_up_is_rewritten_as_standalone(patch_model) -> None:
    """'What about food?' retrieves nothing on its own — the meaning lives in
    the previous turn."""
    patch_model(["What is the returns policy for food?"])
    state = AgentState(
        question="What about food?",
        original_question="What about food?",
        history=[
            ("user", "How long to return an electrical item?"),
            ("assistant", "30 days with a receipt."),
        ],
    )
    result = nodes.contextualize_question(state)

    assert result["question"] == "What is the returns policy for food?"
    assert result["original_question"] == "What is the returns policy for food?"
    assert result["contextualized"] is True


def test_already_standalone_question_is_left_alone(patch_model) -> None:
    patch_model(["How many points for a voucher?"])
    state = AgentState(
        question="How many points for a voucher?",
        history=[("user", "hi"), ("assistant", "hello")],
    )
    assert nodes.contextualize_question(state)["contextualized"] is False


def test_unusable_rewrite_falls_back_to_the_original(patch_model) -> None:
    """A model that answers instead of rewriting would poison retrieval."""
    patch_model(["   "])
    state = AgentState(question="What about food?", history=[("user", "returns?")])
    result = nodes.contextualize_question(state)

    assert result["contextualized"] is False
    assert "question" not in result  # original left untouched


def test_only_recent_history_is_sent(patch_model) -> None:
    """An unbounded transcript grows prompt cost without bound."""
    model = patch_model(["standalone"])
    history = [("user", f"turn {i}") for i in range(20)]
    nodes.contextualize_question(AgentState(question="and that?", history=history))

    sent = model.calls[0][-1].content
    assert "turn 19" in sent
    assert "turn 0" not in sent


# --- routing --------------------------------------------------------------
def test_router_parses_a_valid_route(patch_model) -> None:
    patch_model(["web_search"])
    result = nodes.route_question(AgentState(question="what is the news?"))
    assert result["route"] is Route.WEB_SEARCH


def test_router_falls_back_to_retrieval_on_garbage(patch_model) -> None:
    """An unparseable route must degrade, not crash."""
    patch_model(["I think you should probably check the internal docs"])
    result = nodes.route_question(AgentState(question="returns?"))
    assert result["route"] is Route.VECTORSTORE


# --- document grading -----------------------------------------------------
def test_grader_discards_irrelevant_chunks(patch_model) -> None:
    """The core of self-RAG: don't trust the retriever blindly."""
    patch_model(["yes", "no", "yes"])
    state = AgentState(
        question="returns?",
        documents=[Document(page_content=str(i)) for i in range(3)],
    )
    result = nodes.grade_documents(state)
    assert len(result["documents"]) == 2
    assert "kept 2/3" in result["trace"][-1]


def test_grader_can_reject_everything(patch_model) -> None:
    patch_model(default="no")
    state = AgentState(question="q", documents=[Document(page_content="unrelated")])
    assert nodes.grade_documents(state)["documents"] == []


# --- rewriting ------------------------------------------------------------
def test_rewrite_uses_the_original_question(patch_model) -> None:
    """Rewriting a rewrite compounds drift away from what was asked."""
    model = patch_model(["electrical item return window policy"])
    state = AgentState(
        question="a previously mangled rewrite",
        original_question="how long to return a kettle?",
        rewrites=1,
    )
    result = nodes.rewrite_query(state)

    assert result["rewrites"] == 2
    sent = model.calls[0][-1].content
    assert sent == "how long to return a kettle?"


# --- generation -----------------------------------------------------------
def test_generate_builds_cited_context(patch_model) -> None:
    patch_model(["Electrical items: 30 days [returns_policy.md]"])
    state = AgentState(
        question="q",
        original_question="q",
        documents=[Document(page_content="30 days", metadata={"source": "returns_policy.md"})],
    )
    result = nodes.generate(state)

    assert "[returns_policy.md]" in result["context"]
    assert result["generation_attempts"] == 1


def test_generate_handles_empty_context(patch_model) -> None:
    """No relevant chunks must still produce an honest answer, not a crash."""
    patch_model(["I don't have that information."])
    result = nodes.generate(AgentState(question="q", original_question="q", documents=[]))
    assert result["context"] == nodes.NO_CONTEXT_SENTINEL


def test_repeat_generation_is_labelled_a_retry(patch_model) -> None:
    patch_model(["second attempt"])
    state = AgentState(question="q", original_question="q", documents=[], generation_attempts=1)
    result = nodes.generate(state)
    assert result["generation_attempts"] == 2
    assert "regenerated" in result["trace"][-1]


# --- generation grading ---------------------------------------------------
def test_ungrounded_answer_is_caught(patch_model) -> None:
    """The hallucination gate is the safety guarantee of the whole system."""
    patch_model(["no"])
    state = AgentState(context="Electrical: 30 days.", generation="You have 90 days.")
    assert nodes.grade_generation(state) is GenerationVerdict.NOT_GROUNDED


def test_grounded_but_unhelpful_answer_is_caught(patch_model) -> None:
    patch_model(["yes", "no"])  # grounded, then not useful
    state = AgentState(
        context="Electrical: 30 days.",
        generation="Returns exist.",
        original_question="how long?",
    )
    assert nodes.grade_generation(state) is GenerationVerdict.NOT_USEFUL


def test_good_answer_passes_both_gates(patch_model) -> None:
    patch_model(["yes", "yes"])
    state = AgentState(
        context="Electrical: 30 days.",
        generation="30 days.",
        original_question="how long?",
    )
    assert nodes.grade_generation(state) is GenerationVerdict.USEFUL


def test_direct_answers_skip_grading(patch_model) -> None:
    """There is no context to check a conversational reply against."""
    model = patch_model(default="no")
    state = AgentState(context=nodes.DIRECT_ANSWER_SENTINEL, generation="Hello!")
    assert nodes.grade_generation(state) is GenerationVerdict.USEFUL
    assert model.calls == []  # no LLM call was wasted
