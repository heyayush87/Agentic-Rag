"""Typed exception hierarchy.

Every error the platform raises deliberately descends from `RetailIQError`,
so the API layer can map "our" failures to clean 4xx/5xx responses while
letting genuinely unexpected exceptions bubble up as 500s with a trace.
"""

from __future__ import annotations

from typing import Any


class RetailIQError(Exception):
    """Base class for every error this platform raises on purpose."""

    #: Default HTTP status the API layer uses when translating this error.
    status_code: int = 500
    #: Stable machine-readable code for clients and log queries.
    code: str = "internal_error"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        """Serialise for an API error body or a structured log record."""
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.context:
            payload["context"] = self.context
        return payload


class ConfigurationError(RetailIQError):
    """Settings are missing, malformed, or mutually inconsistent."""

    status_code = 500
    code = "configuration_error"


class ProviderNotInstalledError(ConfigurationError):
    """A provider was selected whose optional dependency is not installed.

    Raised instead of a bare ``ImportError`` so the message can name the
    exact extra to install rather than leaving the user to decode a
    missing-module traceback.
    """

    code = "provider_not_installed"

    def __init__(self, provider: str, package: str, extra: str) -> None:
        super().__init__(
            f"Provider {provider!r} requires the {package!r} package, which is not installed. "
            f"Install it with:  pip install 'retailiq[{extra}]'",
            provider=provider,
            package=package,
            extra=extra,
        )


class IngestionError(RetailIQError):
    """The knowledge base could not be loaded, chunked, or embedded."""

    status_code = 500
    code = "ingestion_error"


class VectorStoreNotFoundError(RetailIQError):
    """Query-time lookup found no persisted index on disk."""

    status_code = 503
    code = "vector_store_not_found"

    def __init__(self, path: str) -> None:
        super().__init__(
            f"No vector store at {path!r}. Build it first with:  retailiq ingest --reset",
            path=path,
        )


class RetrievalError(RetailIQError):
    """The vector store was reachable but the query failed."""

    status_code = 503
    code = "retrieval_error"


class ProviderQuotaError(RetailIQError):
    """The LLM provider refused the call for quota or rate-limit reasons.

    Worth its own type because the remedy is completely different from a bug:
    nothing is broken, the account is simply out of budget. Surfacing it as a
    generic failure sends people debugging code that is working correctly.

    The agent makes six or seven calls per question, so free tiers are reached
    far sooner here than in a single-shot chatbot — which is exactly why the
    message names the switch-provider fix.
    """

    status_code = 429
    code = "provider_quota_exceeded"

    def __init__(self, provider: str, detail: str = "") -> None:
        super().__init__(
            f"Provider {provider!r} rejected the request: out of quota or rate-limited. "
            f"This agent makes ~7 model calls per question, so free tiers run out quickly. "
            f"Switch provider in .env (e.g. LLM_PROVIDER=groq) or wait for the quota to reset."
            + (f" Provider said: {detail}" if detail else ""),
            provider=provider,
        )


#: Substrings that identify a quota/rate-limit refusal across providers.
#: Matched against the exception text because each SDK raises its own type,
#: and importing all of them here would defeat the point of the factory.
_QUOTA_MARKERS = (
    "402",
    "429",
    "payment required",
    "too many requests",
    "rate limit",
    "quota",
    "depleted",
    "insufficient_quota",
    "credits",
    "billing",
)


def is_quota_error(exc: BaseException) -> bool:
    """Whether `exc` looks like a provider quota or rate-limit refusal."""
    text = str(exc).lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


class AgentExecutionError(RetailIQError):
    """The agent graph failed or exceeded its step budget."""

    status_code = 500
    code = "agent_execution_error"


class ToolExecutionError(RetailIQError):
    """An external tool failed.

    Usually non-fatal: the agent degrades to answering without that tool
    rather than aborting the whole run.
    """

    status_code = 502
    code = "tool_execution_error"


class EvaluationError(RetailIQError):
    """The evaluation harness could not run or its dataset is invalid."""

    status_code = 500
    code = "evaluation_error"
