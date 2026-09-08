"""FastAPI application factory.

    uvicorn retailiq.api.main:app --reload

A factory rather than a module-level app so tests can build an isolated
instance with overridden dependencies.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from retailiq import __version__
from retailiq.api.middleware import CorrelationIdMiddleware, register_exception_handlers
from retailiq.api.routers import admin, health, query
from retailiq.core.logging import configure_logging, get_logger
from retailiq.core.settings import get_settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm up on startup so the first user request is not the slow one.

    Loading the embedding model takes seconds. Doing it here moves that cost
    off the critical path and, just as usefully, surfaces a misconfiguration
    at boot rather than on someone's first question.
    """
    settings = get_settings()
    logger.info("Starting RetailIQ API", extra=settings.describe())

    from retailiq.ingestion.vector_store import index_exists

    if not index_exists(settings):
        logger.warning(
            "No vector index found — queries will fail until it is built. "
            "Run `retailiq ingest --reset` or POST /admin/ingest."
        )
    yield
    logger.info("Shutting down RetailIQ API")


def create_app() -> FastAPI:
    """Build the application."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(
        title="RetailIQ API",
        version=__version__,
        summary="Self-correcting agentic RAG over retail operations knowledge.",
        description=(
            "Ask questions about returns, loyalty, store operations, supplier "
            "logistics and the product catalogue. Every answer reports the "
            "sources it is grounded in and the agent's decision trace."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(CorrelationIdMiddleware)
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(admin.router)

    return app


app = create_app()
