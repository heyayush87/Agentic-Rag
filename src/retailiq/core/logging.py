"""Structured logging with request correlation.

`print()` is fine for a demo and useless in production: you cannot filter it,
ship it to a log aggregator, or tie a line back to the request that produced
it. This module gives every log record a `correlation_id` drawn from a
`contextvar`, so one API request's agent steps can be grepped out of an
interleaved multi-request log.

Two formats, selected by ``LOG_FORMAT``:

* ``console`` — aligned and readable during development.
* ``json``    — one JSON object per line for Datadog / CloudWatch / Loki.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

from retailiq.core.enums import LogFormat

#: Correlation id for the in-flight request. Set by API middleware or the CLI.
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

_CONFIGURED = False

# Attributes present on every LogRecord; anything else was passed via
# `extra=` and is therefore structured context worth emitting.
_RESERVED = frozenset(
    [
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
        "correlation_id",
    ]
)


def new_correlation_id() -> str:
    """Generate and install a fresh correlation id for this context."""
    cid = uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str | None) -> None:
    _correlation_id.set(cid)


def get_correlation_id() -> str | None:
    return _correlation_id.get()


class _CorrelationFilter(logging.Filter):
    """Attach the current correlation id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line, including any `extra=` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable format with the correlation id inline."""

    _FMT = "%(asctime)s %(levelname)-7s [%(correlation_id)s] %(name)s: %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt="%H:%M:%S")


def configure_logging(
    level: str = "INFO",
    fmt: LogFormat = LogFormat.CONSOLE,
    *,
    force: bool = False,
) -> None:
    """Install handlers on the root logger. Idempotent unless ``force``.

    Safe to call from the CLI, the API startup hook, and Streamlit without
    stacking duplicate handlers (which is how you end up with every line
    printed three times).
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt is LogFormat.JSON else ConsoleFormatter())
    handler.addFilter(_CorrelationFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These libraries are extremely chatty at INFO and drown out our own lines.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.LoggerAdapter[logging.Logger] | logging.Logger:
    """Return a module logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)
