"""Ingestion orchestration — the one call the CLI and API admin route share."""

from __future__ import annotations

import time
from dataclasses import dataclass

from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings
from retailiq.ingestion.chunking import chunk_documents
from retailiq.ingestion.loaders import load_knowledge_base
from retailiq.ingestion.vector_store import create_vector_store, drop_index

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Summary of an index build, for CLI output and API responses."""

    documents: int
    chunks: int
    duration_seconds: float
    index_path: str

    def describe(self) -> str:
        return (
            f"Ingested {self.documents} documents into {self.chunks} chunks "
            f"in {self.duration_seconds:.1f}s → {self.index_path}"
        )


def build_index(reset: bool = True, settings: Settings | None = None) -> IngestionResult:
    """Run the full pipeline: load → chunk → embed → persist.

    Args:
        reset: Drop any existing index first. Default True because Chroma
            appends rather than replaces — re-running without a reset
            duplicates every chunk, which quietly degrades retrieval by
            filling the top-k with copies of the same passage.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    if reset:
        drop_index(settings)

    documents = load_knowledge_base(settings)
    chunks = chunk_documents(documents, settings)
    create_vector_store(chunks, settings)

    result = IngestionResult(
        documents=len(documents),
        chunks=len(chunks),
        duration_seconds=time.perf_counter() - started,
        index_path=str(settings.paths.chroma_dir),
    )
    logger.info("Ingestion complete", extra=result.__dict__)
    return result
