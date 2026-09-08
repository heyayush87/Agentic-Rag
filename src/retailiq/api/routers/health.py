"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from retailiq import __version__
from retailiq.api.dependencies import rag_service_provider
from retailiq.api.schemas import HealthResponse
from retailiq.services.rag_service import RAGService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Readiness probe")
def health(service: RAGService = Depends(rag_service_provider)) -> HealthResponse:
    """Report whether the service can actually answer questions.

    Deliberately distinguishes *up* from *ready*: the process can be running
    fine while the vector index is missing, in which case every query would
    fail. Reporting "degraded" lets an orchestrator hold traffic back instead
    of routing users into errors.
    """
    details = service.health()
    index_built = bool(details.get("index_built"))

    return HealthResponse(
        status="ok" if index_built else "degraded",
        version=__version__,
        index_built=index_built,
        details=details,
    )


@router.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": "RetailIQ", "version": __version__, "docs": "/docs"}
