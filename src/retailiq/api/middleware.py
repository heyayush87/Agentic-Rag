"""Cross-cutting HTTP concerns: correlation ids, access logs, error mapping."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from retailiq.core.exceptions import RetailIQError
from retailiq.core.logging import get_logger, new_correlation_id, set_correlation_id

logger = get_logger(__name__)

CORRELATION_HEADER = "X-Correlation-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Give every request a traceable id and log its outcome.

    Honours an inbound `X-Correlation-ID` so a trace can span services, and
    echoes it back on the response so a client can quote it in a bug report.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable]
    ) -> JSONResponse:
        inbound = request.headers.get(CORRELATION_HEADER)
        if inbound:
            set_correlation_id(inbound)
            correlation_id = inbound
        else:
            correlation_id = new_correlation_id()

        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - started) * 1000)

        response.headers[CORRELATION_HEADER] = correlation_id
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """Map platform errors to clean responses.

    Our own exceptions already carry an accurate status code and a stable
    machine-readable `code`, so they become structured 4xx/5xx bodies.
    Anything else is genuinely unexpected: log the traceback, return a
    generic 500, and never leak internals to the caller.
    """
    from retailiq.core.logging import get_correlation_id

    @app.exception_handler(RetailIQError)
    async def _handle_known(_: Request, exc: RetailIQError) -> JSONResponse:
        logger.warning(
            "Handled platform error",
            extra={"code": exc.code, "detail": exc.message},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={**exc.to_dict(), "correlation_id": get_correlation_id()},
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error", extra={"error": type(exc).__name__})
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "An unexpected error occurred.",
                "correlation_id": get_correlation_id(),
            },
        )
