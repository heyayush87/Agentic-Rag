"""Closed vocabularies used across the platform.

These are ``str`` enums so they compare cleanly against raw strings coming
from environment variables and JSON payloads, while still giving the type
checker a finite set of valid values.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """A string enum whose ``str()`` is the bare value, not ``Class.MEMBER``."""

    def __str__(self) -> str:
        return str(self.value)


class AppEnvironment(StrEnum):
    """Deployment tier. Drives log format and error verbosity defaults."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(StrEnum):
    """Supported chat-completion backends."""

    OPENAI = "openai"
    GROQ = "groq"
    GOOGLE = "google"
    OLLAMA = "ollama"


class EmbeddingProvider(StrEnum):
    """Supported embedding backends.

    ``LOCAL`` runs sentence-transformers on-device, which is why Groq (which
    exposes no embedding endpoint) is still a complete configuration.
    """

    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"
    LOCAL = "local"


class LogFormat(StrEnum):
    """``CONSOLE`` for humans, ``JSON`` for log aggregators."""

    CONSOLE = "console"
    JSON = "json"


class Route(StrEnum):
    """Where the router decided a question should be answered from."""

    VECTORSTORE = "vectorstore"
    WEB_SEARCH = "web_search"
    DIRECT_ANSWER = "answer"


class GenerationVerdict(StrEnum):
    """Outcome of the post-generation quality gate."""

    USEFUL = "useful"
    NOT_GROUNDED = "not_grounded"
    NOT_USEFUL = "not_useful"
