"""Tavily web search.

Tavily is built for LLM consumption rather than human browsing: it returns
cleaned, de-duplicated content extracts instead of the title-and-snippet
fragments a conventional SERP gives you. That matters here because whatever
this returns is pasted straight into the generation context — snippet noise
becomes answer noise.

Costs an API key (free tier: 1,000 searches/month). DuckDuckGo remains the
zero-config default for anyone who doesn't want another account.
"""

from __future__ import annotations

from typing import Any

from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings
from retailiq.tools.base import Tool, ToolResult

logger = get_logger(__name__)


class TavilySearchTool(Tool):
    """LLM-optimised web search. Requires `TAVILY_API_KEY`."""

    name = "tavily_search"
    description = (
        "Search the public web for current or external information, returning "
        "cleaned content extracts suitable for grounding an answer."
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def is_available(self) -> bool:
        if not self._settings.tools.enable_web_search:
            return False
        if not self._settings.tools.tavily_api_key_value:
            return False
        try:
            import tavily  # noqa: F401
        except ImportError:
            return False
        return True

    def run(self, query: str, **kwargs: Any) -> ToolResult:
        if not self._settings.tools.enable_web_search:
            return ToolResult.failure(self.name, "disabled by configuration")

        api_key = self._settings.tools.tavily_api_key_value
        if not api_key:
            return ToolResult.failure(self.name, "TAVILY_API_KEY is not set")

        try:
            from tavily import TavilyClient
        except ImportError:
            return ToolResult.failure(
                self.name, "tavily-python not installed (pip install 'retailiq[tavily]')"
            )

        max_results = int(kwargs.get("max_results", self._settings.tools.web_search_max_results))

        try:
            response = TavilyClient(api_key=api_key).search(
                query=query,
                max_results=max_results,
                # "advanced" spends an extra credit to return longer, cleaner
                # extracts. Worth it when the result is grounding an answer.
                search_depth="advanced",
            )
        except Exception as exc:
            logger.warning("Tavily search failed", extra={"error": type(exc).__name__})
            return ToolResult.failure(self.name, type(exc).__name__)

        results = response.get("results", []) if isinstance(response, dict) else []
        if not results:
            return ToolResult(
                content="[no web results found]",
                tool_name=self.name,
                metadata={"results": 0},
            )

        blocks = [
            f"{row.get('title', '')}\n{row.get('content', '')}\n(source: {row.get('url', '')})"
            for row in results
        ]
        return ToolResult(
            content="\n\n".join(blocks),
            tool_name=self.name,
            metadata={"results": len(blocks), "provider": "tavily"},
        )
