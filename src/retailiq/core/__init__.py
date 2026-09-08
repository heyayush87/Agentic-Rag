"""Cross-cutting concerns: configuration, logging, error types, enums.

Nothing in this package may import from the feature packages (agent,
ingestion, llm, services, api). The dependency arrow points one way only —
features depend on core, never the reverse.
"""

from __future__ import annotations

from retailiq.core.enums import (
    AppEnvironment,
    EmbeddingProvider,
    LLMProvider,
    LogFormat,
    Route,
)
from retailiq.core.exceptions import (
    AgentExecutionError,
    ConfigurationError,
    IngestionError,
    ProviderNotInstalledError,
    RetailIQError,
    ToolExecutionError,
    VectorStoreNotFoundError,
)
from retailiq.core.settings import Settings, get_settings

__all__ = [
    "AgentExecutionError",
    "AppEnvironment",
    "ConfigurationError",
    "EmbeddingProvider",
    "IngestionError",
    "LLMProvider",
    "LogFormat",
    "ProviderNotInstalledError",
    "RetailIQError",
    "Route",
    "Settings",
    "ToolExecutionError",
    "VectorStoreNotFoundError",
    "get_settings",
]
