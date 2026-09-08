"""Operational endpoints: index rebuild and evaluation.

Unauthenticated in this reference build. Before any real deployment these
must sit behind auth — `/admin/ingest` is expensive and destructive, and
`/admin/evaluate` spends provider tokens on every call.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from retailiq.api.dependencies import evaluation_service_provider, reset_service_cache
from retailiq.api.schemas import IngestRequest, IngestResponse
from retailiq.ingestion.pipeline import build_index
from retailiq.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest", response_model=IngestResponse, summary="Rebuild the vector index")
def ingest(payload: IngestRequest) -> IngestResponse:
    """Re-run the ingestion pipeline over the knowledge base."""
    result = build_index(reset=payload.reset)

    # The rebuild replaced the files the cached store handle points at, so
    # drop the cached services or queries would keep reading the old index.
    reset_service_cache()

    return IngestResponse(
        documents=result.documents,
        chunks=result.chunks,
        duration_seconds=round(result.duration_seconds, 2),
        index_path=result.index_path,
    )


@router.post("/evaluate", summary="Run the evaluation harness")
def evaluate(
    service: EvaluationService = Depends(evaluation_service_provider),
) -> dict[str, object]:
    """Score the assistant against the golden dataset.

    Synchronous and slow — it runs the full agent once per case. Fine for a
    manual quality check or a nightly job; for anything user-facing this
    belongs on a task queue.
    """
    report = service.run()
    return {
        "summary": report.summary(),
        "rows": [
            {
                "question": row.question,
                "retrieved_expected_source": row.retrieved_expected_source,
                "keywords_present": row.keywords_present,
                "faithful": row.faithful,
                "relevant": row.relevant,
                "passed": row.passed,
            }
            for row in report.rows
        ],
    }
