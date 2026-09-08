"""Shared fixtures.

Guiding rule: unit and integration tests never touch the network or a real
LLM. A test suite that needs an API key is a test suite nobody runs in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from retailiq.core.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's real `.env` from changing test outcomes."""
    for var in (
        "LLM_PROVIDER",
        "EMBED_PROVIDER",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVE_K",
        "MAX_QUERY_REWRITES",
        "ENABLE_WEB_SEARCH",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a temporary directory, never the real index."""
    return Settings(
        paths={"data_dir": tmp_path / "data", "chroma_dir": tmp_path / "chroma"},  # type: ignore[arg-type]
    )


@pytest.fixture
def knowledge_base(tmp_path: Path) -> Path:
    """A miniature knowledge base on disk."""
    directory = tmp_path / "data" / "knowledge_base"
    directory.mkdir(parents=True)
    (directory / "returns_policy.md").write_text(
        "## Electrical returns\n"
        "Electrical items may be returned within 30 days with a receipt.\n\n"
        "## Food returns\n"
        "Food cannot be returned on a change of mind for hygiene reasons.\n",
        encoding="utf-8",
    )
    (directory / "loyalty_clubcard.md").write_text(
        "## Vouchers\nYou need 150 points for a voucher. Vouchers last 2 years.\n",
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def eval_dataset(tmp_path: Path) -> Path:
    """A valid two-case evaluation dataset."""
    directory = tmp_path / "data" / "eval"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "eval_questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "How long to return an electrical item?",
                    "expected_source": "returns_policy.md",
                    "must_include": ["30 days"],
                },
                {
                    "question": "How many points for a voucher?",
                    "expected_source": "loyalty_clubcard.md",
                    "must_include": ["150"],
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


class FakeMessage:
    """Minimal stand-in for a LangChain message response."""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatModel:
    """Scripted chat model. Returns queued replies in order, then a default.

    Lets the agent's control flow be tested deterministically — "what does
    the graph do when the grader says no twice?" — with no provider, no key
    and no latency.
    """

    def __init__(self, replies: list[str] | None = None, default: str = "yes") -> None:
        self.replies = list(replies or [])
        self.default = default
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> FakeMessage:
        self.calls.append(messages)
        return FakeMessage(self.replies.pop(0) if self.replies else self.default)


@pytest.fixture
def fake_chat_model() -> type[FakeChatModel]:
    return FakeChatModel
