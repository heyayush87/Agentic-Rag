"""Domain model behaviour: context rendering, sources, metric aggregation."""

from __future__ import annotations

import pytest

from retailiq.domain.models import (
    EvaluationReport,
    EvaluationRowResult,
    QueryResult,
    RetrievedChunk,
)

pytestmark = pytest.mark.unit


def _chunk(source: str, content: str = "text") -> RetrievedChunk:
    return RetrievedChunk(source=source, topic=source.split(".")[0], content=content)


def test_context_tags_every_chunk_with_its_source() -> None:
    result = QueryResult(
        question="q",
        answer="a",
        chunks=[_chunk("returns_policy.md", "30 days"), _chunk("loyalty.md", "150 points")],
    )
    assert "[returns_policy.md]" in result.context
    assert "[loyalty.md]" in result.context


def test_empty_context_uses_the_sentinel() -> None:
    assert QueryResult(question="q", answer="a").context == "(no context retrieved)"


def test_sources_are_deduplicated_in_order() -> None:
    result = QueryResult(
        question="q",
        answer="a",
        chunks=[_chunk("b.md"), _chunk("a.md"), _chunk("b.md")],
    )
    assert result.sources == ["b.md", "a.md"]


def test_a_row_passes_only_when_every_metric_passes() -> None:
    kwargs = {
        "question": "q",
        "retrieved_expected_source": True,
        "keywords_present": True,
        "faithful": True,
        "relevant": True,
    }
    assert EvaluationRowResult(**kwargs).passed is True
    assert EvaluationRowResult(**{**kwargs, "faithful": False}).passed is False


def test_report_aggregates_each_metric_independently() -> None:
    """Separate metrics localise a regression to retrieval or to generation."""
    report = EvaluationReport(
        rows=[
            EvaluationRowResult(
                question="1",
                retrieved_expected_source=True,
                keywords_present=True,
                faithful=True,
                relevant=True,
            ),
            EvaluationRowResult(
                question="2",
                retrieved_expected_source=True,
                keywords_present=False,
                faithful=True,
                relevant=False,
            ),
        ]
    )
    assert report.total == 2
    assert report.retrieval_hit_rate == 1.0
    assert report.keyword_recall == 0.5
    assert report.answer_relevance == 0.5


def test_empty_report_does_not_divide_by_zero() -> None:
    assert EvaluationReport().retrieval_hit_rate == 0.0
