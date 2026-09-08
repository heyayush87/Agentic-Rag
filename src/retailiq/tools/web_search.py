"""Web-search fallback for questions the internal knowledge base cannot answer.

Degrades gracefully by design: if the package is missing or the network is
unreachable, it returns a clear stub instead of an exception, so the project
still runs end to end in an offline or locked-down environment.
"""

from __future__ import annotations

from typing import Any

from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings
from retailiq.tools.base import Tool, ToolResult

logger = get_logger(__name__)


class WebSearchTool(Tool):
    """DuckDuckGo search. No API key required."""

    name = "web_search"
    description = (
        "Search the public web for current or general-world information that the "
        "internal retail knowledge base would not contain."
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def is_available(self) -> bool:
        if not self._settings.tools.enable_web_search:
            return False
        try:
            import duckduckgo_search  # noqa: F401
        except ImportError:
            return False
        return True

    def run(self, query: str, **kwargs: Any) -> ToolResult:
        if not self._settings.tools.enable_web_search:
            return ToolResult.failure(self.name, "disabled by configuration")

        max_results = int(kwargs.get("max_results", self._settings.tools.web_search_max_results))

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return ToolResult.failure(
                self.name, "duckduckgo-search not installed (pip install 'retailiq[search]')"
            )

        try:
            snippets: list[str] = []
            with DDGS() as ddgs:
                for row in ddgs.text(query, max_results=max_results):
                    snippets.append(
                        f"{row.get('title', '')}\n"
                        f"{row.get('body', '')}\n"
                        f"(source: {row.get('href', '')})"
                    )
        except Exception as exc:
            logger.warning("Web search failed", extra={"error": type(exc).__name__})
            return ToolResult.failure(self.name, type(exc).__name__)

        if not snippets:
            return ToolResult(
                content="[no web results found]",
                success=True,
                tool_name=self.name,
                metadata={"results": 0},
            )

        return ToolResult(
            content="\n\n".join(snippets),
            tool_name=self.name,
            metadata={"results": len(snippets)},
        )
