"""LangSmith tracing and online evaluation.

Why this exists alongside the test suite
----------------------------------------
They cover different failure classes at different times, and neither
substitutes for the other:

* **pytest (pre-merge, offline)** catches *code* defects — a misconfigured
  alias, an eager-default `KeyError`, an unbounded retry loop. Deterministic,
  free, and it blocks a bad merge before anyone is affected.
* **LangSmith (post-deploy, live)** catches *quality* defects — a prompt
  change that quietly raises the hallucination rate, p95 latency creeping up,
  token spend per question doubling. No unit test can observe these, because
  they only appear against real traffic and a real model.

The agent makes six or seven LLM calls per question. When an answer is wrong,
"which call went wrong?" is unanswerable from logs alone — you need the tree.
That is what tracing provides: the router's choice, each grader's verdict,
the rewrite, and the final generation, nested and timed.

Entirely optional. With no `LANGCHAIN_API_KEY` set, tracing is off and the
platform behaves exactly as before — no network calls, no vendor lock-in.
"""

from __future__ import annotations

import os
from typing import Any

from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings

logger = get_logger(__name__)


def configure_tracing(settings: Settings | None = None) -> bool:
    """Enable LangSmith tracing if it is configured.

    LangChain reads tracing configuration from environment variables at call
    time, so this promotes our validated settings into the variables the SDK
    expects rather than requiring the user to set both.

    Returns:
        True if tracing was enabled, False if it is disabled or unconfigured.
    """
    settings = settings or get_settings()
    observability = settings.observability

    if not observability.tracing_enabled:
        return False

    api_key = observability.api_key_value
    if not api_key:
        logger.warning(
            "LANGCHAIN_TRACING_V2 is on but LANGCHAIN_API_KEY is unset; tracing disabled."
        )
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = observability.project
    os.environ["LANGCHAIN_ENDPOINT"] = observability.endpoint

    logger.info(
        "LangSmith tracing enabled",
        extra={"project": observability.project, "environment": str(settings.environment)},
    )
    return True


def trace_metadata(settings: Settings | None = None, **extra: Any) -> dict[str, Any]:
    """Metadata attached to every traced run.

    Tagging by environment, provider and model is what makes a trace
    *queryable* later: "show me ungrounded answers from production on
    llama-3.3-70b in the last day" is only answerable if these are recorded
    at trace time.
    """
    settings = settings or get_settings()
    return {
        "environment": str(settings.environment),
        "llm_provider": str(settings.llm.provider),
        "embed_provider": str(settings.embeddings.provider),
        "app_version": _version(),
        **extra,
    }


def _version() -> str:
    from retailiq import __version__

    return __version__


def is_tracing_active() -> bool:
    """Whether LangChain will currently emit traces."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
