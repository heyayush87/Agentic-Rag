"""Domain models — the vocabulary of the platform.

Plain pydantic objects with no LangChain, HTTP, or storage types in their
signatures. Both the API layer and the CLI render these, which is what keeps
the transport layers from re-deriving business meaning independently.
"""

from __future__ import annotations

from retailiq.domain.models import (
    ChatTurn,
    Conversation,
    EvaluationCase,
    EvaluationReport,
    EvaluationRowResult,
    QueryResult,
    RetrievedChunk,
    TraceStep,
)

__all__ = [
    "ChatTurn",
    "Conversation",
    "EvaluationCase",
    "EvaluationReport",
    "EvaluationRowResult",
    "QueryResult",
    "RetrievedChunk",
    "TraceStep",
]
