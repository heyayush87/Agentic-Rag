"""Error hierarchy and its mapping to HTTP semantics."""

from __future__ import annotations

import pytest

from retailiq.core.exceptions import (
    ConfigurationError,
    ProviderNotInstalledError,
    RetailIQError,
    VectorStoreNotFoundError,
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


def test_context_is_serialised_for_structured_logs() -> None:
    payload = ConfigurationError("bad", provider="groq").to_dict()
    assert payload["code"] == "configuration_error"
    assert payload["context"] == {"provider": "groq"}
