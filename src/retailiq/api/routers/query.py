"""The question-answering endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from retailiq.api.dependencies import rag_service_provider
from retailiq.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from retailiq.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask the retail assistant a question",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "The vector index has not been built yet.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def query(
    payload: QueryRequest,
    service: RAGService = Depends(rag_service_provider),
) -> QueryResponse:
    """Answer a question with a full account of how the answer was reached.

    The response reports the route taken, the source documents, whether the
    answer passed the grounding check, and how many times the query was
    rewritten — so a caller can decide how much to trust it rather than
    taking the text on faith.
    """
    result = service.answer(payload.question)
    return QueryResponse.from_domain(
        result,
        include_context=payload.include_context,
        include_trace=payload.include_trace,
    )
