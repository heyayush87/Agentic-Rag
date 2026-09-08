"""RetailIQ — enterprise agentic RAG platform for retail operations knowledge.

A self-correcting retrieval system built on LangGraph. Rather than the usual
"retrieve then answer" pipeline, the agent reasons about *how* to answer:
it routes the question, grades its own retrieved documents for relevance,
rewrites weak queries and retries, then verifies the drafted answer is
grounded in the retrieved context before returning it.

Public entry points:
    from retailiq import RAGService, get_settings
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["RAGService", "__version__", "get_settings"]


def __getattr__(name: str) -> object:
    """Lazily expose the public API.

    Importing ``retailiq`` must stay cheap and side-effect free: the CLI's
    ``--help`` should not pay for loading LangChain, Chroma or torch. These
    names resolve on first attribute access instead of at import time.
    """
    if name == "RAGService":
        from retailiq.services.rag_service import RAGService

        return RAGService
    if name == "get_settings":
        from retailiq.core.settings import get_settings

        return get_settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
