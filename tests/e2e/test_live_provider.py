"""End-to-end runs against a real provider and a real index.

Skipped automatically unless a key is configured and the index is built, so
`pytest` stays green on a fresh clone. Run deliberately:

    retailiq ingest --reset && pytest -m e2e
"""

from __future__ import annotations

import pytest

from retailiq.core.settings import get_settings
from retailiq.ingestion.vector_store import index_exists

pytestmark = pytest.mark.e2e


def _skip_reason() -> str | None:
    settings = get_settings()
    if settings.llm.requires_api_key and not settings.llm.api_key_for(settings.llm.provider):
        return f"no API key configured for provider {settings.llm.provider}"
    if not index_exists(settings):
        return "vector index not built (run: retailiq ingest --reset)"
    return None


@pytest.fixture(autouse=True)
def _requires_live_stack() -> None:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)


def test_known_policy_question_is_answered_from_the_right_document() -> None:
    from retailiq.services.rag_service import RAGService

    result = RAGService().answer("How long do I have to return an electrical item?")

    assert "30" in result.answer
    assert "returns_policy.md" in result.sources
    assert result.grounded is True


def test_trace_is_populated() -> None:
    """The decision trace is a product feature, not debug output."""
    from retailiq.services.rag_service import RAGService

    result = RAGService().answer("How many points do I need for a voucher?")
    assert result.trace
    assert any("router" in step.detail for step in result.trace)


def test_out_of_scope_question_is_not_fabricated() -> None:
    """The system must decline rather than invent a policy."""
    from retailiq.services.rag_service import RAGService

    result = RAGService().answer("What is our CEO's home address?")
    assert result.answer
