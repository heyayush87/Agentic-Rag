"""FastAPI dependency providers.

Everything the routes need arrives through `Depends`, which is what lets a
test swap in a stubbed service with `app.dependency_overrides` instead of
monkey-patching module globals.
"""

from __future__ import annotations

from functools import lru_cache

from retailiq.core.settings import Settings, get_settings
from retailiq.services.evaluation_service import EvaluationService
from retailiq.services.rag_service import RAGService


def settings_provider() -> Settings:
    """Injectable handle on configuration."""
    return get_settings()


@lru_cache(maxsize=1)
def _rag_service() -> RAGService:
    return RAGService()


@lru_cache(maxsize=1)
def _evaluation_service() -> EvaluationService:
    return EvaluationService()


def rag_service_provider() -> RAGService:
    """Shared `RAGService`.

    Cached because the underlying graph and embedding model are expensive to
    construct and safe to share — they hold no per-request state.
    """
    return _rag_service()


def evaluation_service_provider() -> EvaluationService:
    return _evaluation_service()


def reset_service_cache() -> None:
    """Drop cached services after a config change or an index rebuild."""
    _rag_service.cache_clear()
    _evaluation_service.cache_clear()
