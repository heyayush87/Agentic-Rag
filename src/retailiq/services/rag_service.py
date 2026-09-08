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
from retailiq.core.exceptions import AgentExecutionError, RetailIQError
from retailiq.core.logging import get_logger, new_correlation_id
from retailiq.core.settings import Settings, get_settings
from retailiq.domain.models import QueryResult, RetrievedChunk, TraceStep

logger = get_logger(__name__)


class RAGService:
    """Answers questions using the agentic RAG graph."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -- public API --------------------------------------------------------
    def answer(self, question: str, *, correlation_id: str | None = None) -> QueryResult:
        """Answer one question.

        Args:
            question: The user's natural-language question.
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

        try:
            final_state = get_compiled_graph().invoke(
                initial_state(question),
                config={"recursion_limit": self._settings.agent.recursion_limit},
            )
        except RetailIQError:
            # Already a typed, well-described failure — don't rewrap and lose it.
            raise
        except Exception as exc:
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
