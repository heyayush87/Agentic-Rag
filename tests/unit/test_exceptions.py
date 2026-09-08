"""Error hierarchy and its mapping to HTTP semantics."""

from __future__ import annotations

import pytest

from retailiq.core.exceptions import (
    ConfigurationError,
    ProviderNotInstalledError,
    ProviderQuotaError,
    RetailIQError,
    VectorStoreNotFoundError,
    is_quota_error,
)

pytestmark = pytest.mark.unit


def test_every_platform_error_shares_a_base() -> None:
    """Lets the API distinguish our failures from genuine bugs."""
    for exc in (
        ConfigurationError("x"),
        VectorStoreNotFoundError("/tmp/chroma"),
        ProviderNotInstalledError("groq", "langchain-groq", "groq"),
    ):
        assert isinstance(exc, RetailIQError)


def test_missing_index_is_503_not_500() -> None:
    """The service is configured correctly, just not ready yet."""
    assert VectorStoreNotFoundError("/tmp/chroma").status_code == 503


def test_missing_index_message_names_the_fix() -> None:
    assert "retailiq ingest" in VectorStoreNotFoundError("/tmp/chroma").message


def test_missing_provider_names_the_exact_install_command() -> None:
    """Better than decoding a raw ImportError traceback."""
    exc = ProviderNotInstalledError("openai", "langchain-openai", "openai")
    assert "pip install 'retailiq[openai]'" in exc.message
    assert isinstance(exc, ConfigurationError)


@pytest.mark.parametrize(
    "message",
    [
        "Client error '402 Payment Required' for url ...",
        "429 Too Many Requests",
        "You have depleted your monthly included credits",
        "RateLimitError: insufficient_quota",
    ],
)
def test_quota_refusals_are_recognised(message: str) -> None:
    """Each provider raises its own type, so detection is on the text."""
    assert is_quota_error(Exception(message)) is True


def test_ordinary_errors_are_not_mistaken_for_quota() -> None:
    assert is_quota_error(ValueError("index out of range")) is False
    assert is_quota_error(KeyError("question")) is False


def test_quota_error_names_the_fix_and_is_429() -> None:
    """Out of budget is not a malfunction; the message must say what to do."""
    exc = ProviderQuotaError("huggingface", "depleted credits")
    assert exc.status_code == 429
    assert exc.code == "provider_quota_exceeded"
    assert "LLM_PROVIDER=groq" in exc.message


def test_context_is_serialised_for_structured_logs() -> None:
    payload = ConfigurationError("bad", provider="groq").to_dict()
    assert payload["code"] == "configuration_error"
    assert payload["context"] == {"provider": "groq"}
