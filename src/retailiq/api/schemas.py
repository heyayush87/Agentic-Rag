"""Request/response contracts for the REST API.

Separate from `domain.models` on purpose: the domain model is free to change
shape internally, while these are a published contract with clients. The
mapping between them is explicit and one directional.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from retailiq.core.enums import Route
from retailiq.domain.models import QueryResult


class QueryRequest(BaseModel):
    """A question submitted to the assistant."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"question": "How long do I have to return an electrical item?"}
        }
    )

    question: str = Field(min_length=1, max_length=2000, description="Natural-language question.")
    include_context: bool = Field(
        default=False, description="Return the retrieved chunks alongside the answer."
    )
    include_trace: bool = Field(default=True, description="Return the agent's decision trace.")


class TraceStepResponse(BaseModel):
    step: str
    detail: str


class ChunkResponse(BaseModel):
    source: str
    topic: str
    content: str


class QueryResponse(BaseModel):
    """The assistant's answer plus the reasoning that produced it."""

    question: str
    answer: str
    route: Route
    sources: list[str] = Field(description="Documents the answer is grounded in.")
    grounded: bool = Field(description="Passed the hallucination check.")
    rewrites: int
    latency_ms: int
    created_at: datetime
    trace: list[TraceStepResponse] | None = None
    chunks: list[ChunkResponse] | None = None

    @classmethod
    def from_domain(
        cls, result: QueryResult, *, include_context: bool, include_trace: bool
    ) -> QueryResponse:
        return cls(
            question=result.question,
            answer=result.answer,
            route=result.route,
            sources=result.sources,
            grounded=result.grounded,
            rewrites=result.rewrites,
            latency_ms=result.latency_ms,
            created_at=result.created_at,
            trace=(
                [TraceStepResponse(step=s.step, detail=s.detail) for s in result.trace]
                if include_trace
                else None
            ),
            chunks=(
                [
                    ChunkResponse(source=c.source, topic=c.topic, content=c.content)
                    for c in result.chunks
                ]
                if include_context
                else None
            ),
        )


class HealthResponse(BaseModel):
    """Liveness and readiness signal."""

    status: str = Field(description="'ok' when ready to serve, 'degraded' otherwise.")
    version: str
    index_built: bool
    details: dict[str, object] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    reset: bool = Field(
        default=True,
        description="Drop the existing index first. Without it, Chroma appends and duplicates every chunk.",
    )


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    duration_seconds: float
    index_path: str


class ErrorResponse(BaseModel):
    """Uniform error body for every failure the platform raises on purpose."""

    code: str
    message: str
    context: dict[str, object] = Field(default_factory=dict)
    correlation_id: str | None = None
