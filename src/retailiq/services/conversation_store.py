"""Durable conversation storage.

Backs the "recent chats" list. Deliberately *not* Streamlit session state:
session state dies with the browser tab, so a conversation history that lived
there would vanish on every refresh — which is not a history, it's a buffer.

One JSON file per conversation under ``var/conversations/``. That choice keeps
the platform dependency-free and the data trivially inspectable; the class
boundary is what matters, so swapping in SQLite or Postgres later means
rewriting this file and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings
from retailiq.domain.models import Conversation

logger = get_logger(__name__)


class ConversationStore:
    """Persists conversations as JSON files, newest-first by update time."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._directory = Path(settings.paths.conversations_dir)
        self._directory.mkdir(parents=True, exist_ok=True)

    # -- writes ------------------------------------------------------------
    def save(self, conversation: Conversation) -> None:
        """Write a conversation, atomically.

        Writes to a temporary file then replaces, so a crash mid-write cannot
        leave a truncated file that fails to parse on next load.
        """
        path = self._path(conversation.id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(conversation.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    def delete(self, conversation_id: str) -> None:
        self._path(conversation_id).unlink(missing_ok=True)

    def delete_all(self) -> int:
        removed = 0
        for path in self._directory.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    # -- reads -------------------------------------------------------------
    def get(self, conversation_id: str) -> Conversation | None:
        path = self._path(conversation_id)
        if not path.exists():
            return None
        return self._load(path)

    def list_recent(self, limit: int = 50) -> list[Conversation]:
        """Conversations ordered most-recently-updated first."""
        conversations = [
            loaded
            for path in self._directory.glob("*.json")
            if (loaded := self._load(path)) is not None
        ]
        conversations.sort(key=lambda c: c.updated_at, reverse=True)
        return conversations[:limit]

    # -- internals ---------------------------------------------------------
    def _path(self, conversation_id: str) -> Path:
        # Guard against a crafted id escaping the store directory.
        safe = "".join(ch for ch in conversation_id if ch.isalnum() or ch in "-_")
        return self._directory / f"{safe}.json"

    def _load(self, path: Path) -> Conversation | None:
        """Read one conversation, skipping anything unreadable.

        A single corrupt file must not take down the whole sidebar, so a
        parse failure is logged and dropped rather than raised.
        """
        try:
            return Conversation(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning(
                "Skipping unreadable conversation file",
                extra={"path": path.name, "error": type(exc).__name__},
            )
            return None
