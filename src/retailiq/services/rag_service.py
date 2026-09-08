"""The primary use case: answer a question.

Translates between the graph's loose `AgentState` dict and the typed
`QueryResult` the outside world consumes. Every transport shares this, so an
answer means the same thing over HTTP as it does in the terminal.
"""

from __future__ import annotations

import time

from langchain_core.documents import Document

from retailiq.agent.graph import get_compiled_graph
from retailiq.agent.state import initial_state
from retailiq.core.enums import Route
from retailiq.core.exceptions import (
    AgentExecutionError,
    ProviderQuotaError,
    RetailIQError,
    is_quota_error,
)
from retailiq.core.logging import get_logger, new_correlation_id
from retailiq.core.settings import Settings, get_settings
from retailiq.domain.models import ChatTurn, QueryResult, RetrievedChunk, TraceStep

logger = get_logger(__name__)


class RAGService:
    """Answers questions using the agentic RAG graph."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -- public API --------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        history: list[ChatTurn] | None = None,
        correlation_id: str | None = None,
    ) -> QueryResult:
        """Answer one question, optionally in the context of a conversation.

        Args:
            question: The user's natural-language question.
            history: Prior turns in this conversation. When supplied, the
                agent resolves references ("what about food?") against them
                before retrieval. Omit it for a one-shot question — the graph
                then skips the contextualisation call entirely.
            correlation_id: Ties this run's logs to a caller's request. One
                is generated if not supplied.

        Raises:
            AgentExecutionError: The graph failed or hit its step limit.
            RetailIQError: Propagated as-is (e.g. the index is not built),
                because those carry their own accurate status codes.
        """
        question = question.strip()
        if not question:
            raise AgentExecutionError("Question must not be empty.")

        if correlation_id:
            from retailiq.core.logging import set_correlation_id

            set_correlation_id(correlation_id)
        else:
            new_correlation_id()

        started = time.perf_counter()
        logger.info("Answering question", extra={"question": question})

        turns = [(turn.role, turn.content) for turn in (history or [])]

        try:
            final_state = get_compiled_graph().invoke(
                initial_state(question, history=turns),
                config={"recursion_limit": self._settings.agent.recursion_limit},
            )
        except RetailIQError:
            # Already a typed, well-described failure — don't rewrap and lose it.
            raise
        except Exception as exc:
            # A quota refusal is not a malfunction, and saying "agent graph
            # failed" sends people debugging code that works.
            if is_quota_error(exc):
                raise ProviderQuotaError(str(self._settings.llm.provider), str(exc)[:200]) from exc
            raise AgentExecutionError(f"Agent graph failed: {exc}", question=question) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        result = self._to_result(question, final_state, latency_ms)

        logger.info(
            "Answered question",
            extra={
                "latency_ms": latency_ms,
                "route": str(result.route),
                "rewrites": result.rewrites,
                "sources": result.sources,
            },
        )
        return result

    def health(self) -> dict[str, object]:
        """Readiness snapshot for the API health endpoint."""
        from retailiq.ingestion.vector_store import index_exists

        return {
            **self._settings.describe(),
            "index_built": index_exists(self._settings),
        }

    def index_stats(self) -> dict[str, object]:
        """Live facts about the built index, for display in a UI.

        Read from the store itself rather than hard-coded, so the numbers
        stay true after a re-ingest or a change to the knowledge base.
        """
        from retailiq.ingestion.loaders import discover_documents
        from retailiq.ingestion.vector_store import get_vector_store, index_exists

        if not index_exists(self._settings):
            return {"built": False, "documents": 0, "chunks": 0, "sources": []}

        try:
            chunks = get_vector_store(self._settings)._collection.count()
        except Exception:
            chunks = 0

        try:
            paths = discover_documents(self._settings.paths.knowledge_base_dir)
        except Exception:
            paths = []

        return {
            "built": True,
            "documents": len(paths),
            "chunks": chunks,
            "sources": [p.name for p in paths],
        }

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _to_result(question: str, state: dict, latency_ms: int) -> QueryResult:
        """Map the final graph state onto the public result model."""
        documents: list[Document] = state.get("documents", []) or []
        chunks = [
            RetrievedChunk(
                source=str(d.metadata.get("source", "unknown")),
                topic=str(d.metadata.get("topic", "")),
                content=d.page_content,
            )
            for d in documents
        ]

        trace = [
            TraceStep(step=entry.split(" ", 1)[0], detail=entry) for entry in state.get("trace", [])
        ]

        context = state.get("context", "")
        return QueryResult(
            question=question,
            answer=state.get("generation") or "(no answer generated)",
            route=Route(state.get("route", Route.VECTORSTORE)),
            chunks=chunks,
            trace=trace,
            rewrites=int(state.get("rewrites", 0)),
            # A run that never produced real context was never graded for
            # grounding, so report it as ungrounded rather than implying a
            # check passed that never ran.
            grounded=bool(context) and not context.startswith("(no"),
            latency_ms=latency_ms,
        )
