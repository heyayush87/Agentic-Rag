"""Configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from retailiq.core.enums import EmbeddingProvider, LLMProvider
from retailiq.core.settings import APISettings, LLMSettings, RetrievalSettings, Settings

pytestmark = pytest.mark.unit


def test_defaults_are_the_free_stack() -> None:
    """Out of the box the project must run on free tiers."""
    settings = Settings()
    assert settings.llm.provider is LLMProvider.GROQ
    assert settings.embeddings.provider is EmbeddingProvider.LOCAL


def test_provider_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "  OpenAI  ")
    assert LLMSettings().provider is LLMProvider.OPENAI


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must fail at startup, not on the first request."""
    monkeypatch.setenv("LLM_PROVIDER", "gpt5-turbo-max")
    with pytest.raises(ValidationError):
        LLMSettings()


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    """An overlap >= chunk size makes the splitter loop; reject it early."""
    with pytest.raises(ValidationError, match="must be smaller than"):
        RetrievalSettings(chunk_size=200, chunk_overlap=200)


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RetrievalSettings(top_k=0)


def test_api_keys_are_not_leaked_by_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key must never appear in a log line or a crash dump."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecret_value")
    settings = LLMSettings()
    assert "gsk_supersecret_value" not in repr(settings)
    assert "gsk_supersecret_value" not in str(settings.groq_api_key)
    # ...but it is still retrievable deliberately.
    assert settings.api_key_for(LLMProvider.GROQ) == "gsk_supersecret_value"


def test_ollama_needs_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert LLMSettings().requires_api_key is False


def test_describe_never_contains_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_leak_me")
    snapshot = Settings().describe()
    assert "gsk_leak_me" not in str(snapshot)
    assert snapshot["api_key_configured"] is True


def test_cors_origins_parsed_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_CORS_ORIGINS", "https://a.com, https://b.com")
    assert APISettings().cors_origins == ["https://a.com", "https://b.com"]


def test_knowledge_base_and_eval_paths_derive_from_data_dir(tmp_path) -> None:
    settings = Settings(paths={"data_dir": tmp_path})  # type: ignore[arg-type]
    assert settings.paths.knowledge_base_dir == tmp_path / "knowledge_base"
    assert settings.paths.eval_dataset == tmp_path / "eval" / "eval_questions.json"
