"""Conversation model and durable storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from retailiq.core.settings import Settings
from retailiq.domain.models import ChatTurn, Conversation
from retailiq.services.conversation_store import ConversationStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path) -> ConversationStore:
    settings = Settings(paths={"chroma_dir": tmp_path / "chroma"})  # type: ignore[arg-type]
    return ConversationStore(settings)


# --- model ----------------------------------------------------------------
def test_title_derives_from_the_first_user_message() -> None:
    """A sidebar full of "New chat" entries is unusable."""
    conversation = Conversation()
    conversation.add(ChatTurn(role="user", content="How long to return a kettle?"))
    assert conversation.title == "How long to return a kettle?"


def test_long_titles_are_truncated() -> None:
    conversation = Conversation()
    conversation.add(ChatTurn(role="user", content="word " * 40))
    assert len(conversation.title) <= 48
    assert conversation.title.endswith("…")


def test_assistant_replies_do_not_rename_the_thread() -> None:
    conversation = Conversation()
    conversation.add(ChatTurn(role="user", content="First question"))
    conversation.add(ChatTurn(role="assistant", content="An answer"))
    assert conversation.title == "First question"


def test_history_excludes_the_message_being_answered() -> None:
    """The agent must not be handed the question it is about to answer as
    'prior context' — it would try to resolve the question against itself."""
    conversation = Conversation()
    conversation.add(ChatTurn(role="user", content="first"))
    conversation.add(ChatTurn(role="assistant", content="reply"))
    conversation.add(ChatTurn(role="user", content="follow-up"))

    history = conversation.history_before_last()
    assert [t.content for t in history] == ["first", "reply"]


def test_empty_conversation_history_is_empty() -> None:
    assert Conversation().history_before_last() == []


# --- persistence ----------------------------------------------------------
def test_conversations_survive_a_new_store_instance(store: ConversationStore) -> None:
    """History must outlive the browser tab — that is the whole point of
    persisting rather than using Streamlit session state."""
    conversation = Conversation()
    conversation.add(ChatTurn(role="user", content="How many points for a voucher?"))
    conversation.add(ChatTurn(role="assistant", content="150.", sources=["loyalty.md"]))
    store.save(conversation)

    reloaded = store.get(conversation.id)
    assert reloaded is not None
    assert reloaded.title == "How many points for a voucher?"
    assert reloaded.turns[1].sources == ["loyalty.md"]


def test_recent_is_ordered_by_update_time(store: ConversationStore) -> None:
    older, newer = Conversation(), Conversation()
    older.add(ChatTurn(role="user", content="older"))
    newer.add(ChatTurn(role="user", content="newer"))

    # Persist out of order to prove sorting is by timestamp, not filesystem.
    store.save(newer)
    store.save(older)
    older.updated_at = older.updated_at.replace(year=2020)
    store.save(older)

    assert [c.title for c in store.list_recent()] == ["newer", "older"]


def test_delete_removes_only_the_named_conversation(store: ConversationStore) -> None:
    keep, drop = Conversation(), Conversation()
    keep.add(ChatTurn(role="user", content="keep"))
    drop.add(ChatTurn(role="user", content="drop"))
    store.save(keep)
    store.save(drop)

    store.delete(drop.id)

    assert store.get(drop.id) is None
    assert store.get(keep.id) is not None


def test_missing_conversation_returns_none(store: ConversationStore) -> None:
    assert store.get("does-not-exist") is None


def test_a_corrupt_file_does_not_break_the_listing(store: ConversationStore) -> None:
    """One bad file must not take down the whole sidebar."""
    good = Conversation()
    good.add(ChatTurn(role="user", content="good"))
    store.save(good)

    (store._directory / "corrupt.json").write_text("{not json", encoding="utf-8")

    recent = store.list_recent()
    assert [c.title for c in recent] == ["good"]


def test_ids_cannot_escape_the_store_directory(store: ConversationStore) -> None:
    """A crafted id must not write outside the conversations folder."""
    path = store._path("../../etc/passwd")
    assert path.parent == store._directory
