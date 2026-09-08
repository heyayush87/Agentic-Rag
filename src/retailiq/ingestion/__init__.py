"""Offline indexing pipeline: load → chunk → embed → persist.

Deliberately separated from query-time code. Ingestion is a batch job with
different failure modes and a different schedule (it runs when documents
change); retrieval is a per-request hot path.
"""

from __future__ import annotations

from retailiq.ingestion.chunking import chunk_documents
from retailiq.ingestion.loaders import load_knowledge_base
from retailiq.ingestion.pipeline import IngestionResult, build_index
from retailiq.ingestion.vector_store import get_retriever, get_vector_store

__all__ = [
    "IngestionResult",
    "build_index",
    "chunk_documents",
    "get_retriever",
    "get_vector_store",
    "load_knowledge_base",
]
