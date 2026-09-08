"""Tool registry — the extension point for new agent capabilities."""

from __future__ import annotations

from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings
from retailiq.tools.base import Tool, ToolResult
from retailiq.tools.web_search import WebSearchTool

logger = get_logger(__name__)


class ToolRegistry:
    """Name-addressed collection of tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool
        logger.debug("Registered tool", extra={"tool": tool.name})

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def run(self, name: str, query: str, **kwargs: object) -> ToolResult:
        """Invoke a tool by name, returning a failure result if unknown."""
        tool = self.get(name)
        if tool is None:
            return ToolResult.failure(name, "tool not registered")
        return tool.run(query, **kwargs)

    def available(self) -> list[Tool]:
        """Tools that can actually run in the current environment."""
        return [t for t in self._tools.values() if t.is_available()]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


def default_registry(settings: Settings | None = None) -> ToolRegistry:
    """Build the registry with the platform's built-in tools."""
    settings = settings or get_settings()
    registry = ToolRegistry()
    registry.register(WebSearchTool(settings))
    return registry
