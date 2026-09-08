"""API contract tests with the service layer stubbed out.

Exercises routing, serialisation and error mapping without an LLM, which is
what makes them runnable in CI on every commit.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from retailiq.api.dependencies import rag_service_provider  # noqa: E402
from retailiq.api.main import create_app  # noqa: E402
from retailiq.core.enums import Route  # noqa: E402
from retailiq.core.exceptions import VectorStoreNotFoundError  # noqa: E402
from retailiq.domain.models import QueryResult, RetrievedChunk, TraceStep  # noqa: E402

pytestmark = pytest.mark.integration


class _StubService:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def answer(self, question: str, **_: object) -> QueryResult:
        if self._error:
            raise self._error
        return QueryResult(
            question=question,
            answer="Electrical items: 30 days [returns_policy.md]",
            route=Route.VECTORSTORE,
            chunks=[RetrievedChunk(source="returns_policy.md", topic="returns", content="30 days")],
            trace=[TraceStep(step="router", detail="router → vectorstore")],
            latency_ms=42,
        )

    def health(self) -> dict[str, object]:
        return {"index_built": self._error is None, "llm_provider": "groq"}


def _client(service: _StubService) -> TestClient:
    app = create_app()
    app.dependency_overrides[rag_service_provider] = lambda: service
    return TestClient(app, raise_server_exceptions=False)


def test_query_returns_answer_and_sources() -> None:
    response = _client(_StubService()).post("/api/v1/query", json={"question": "returns?"})

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == ["returns_policy.md"]
    assert body["route"] == "vectorstore"
    assert body["latency_ms"] == 42


def test_context_is_withheld_unless_requested() -> None:
    client = _client(_StubService())

    assert client.post("/api/v1/query", json={"question": "q"}).json()["chunks"] is None

    with_context = client.post(
        "/api/v1/query", json={"question": "q", "include_context": True}
    ).json()
    assert with_context["chunks"][0]["source"] == "returns_policy.md"


def test_empty_question_is_rejected_by_validation() -> None:
    response = _client(_StubService()).post("/api/v1/query", json={"question": ""})
    assert response.status_code == 422


def test_missing_index_maps_to_503_with_a_stable_code() -> None:
    """Clients should be able to branch on `code`, not parse prose."""
    service = _StubService(error=VectorStoreNotFoundError("/tmp/chroma"))
    response = _client(service).post("/api/v1/query", json={"question": "q"})

    assert response.status_code == 503
    assert response.json()["code"] == "vector_store_not_found"


def test_health_reports_degraded_without_an_index() -> None:
    ok = _client(_StubService()).get("/health").json()
    assert ok["status"] == "ok"

    degraded = _client(_StubService(error=VectorStoreNotFoundError("/x"))).get("/health").json()
    assert degraded["status"] == "degraded"


def test_correlation_id_is_echoed_back() -> None:
    response = _client(_StubService()).get("/health", headers={"X-Correlation-ID": "trace-abc-123"})
    assert response.headers["X-Correlation-ID"] == "trace-abc-123"


def test_openapi_schema_is_served() -> None:
    assert _client(_StubService()).get("/openapi.json").status_code == 200
