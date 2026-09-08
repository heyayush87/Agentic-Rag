"""External tools the agent can invoke.

A registry rather than direct imports: adding a live stock-lookup or order-
status API means writing one `Tool` subclass and registering it, without
editing the agent nodes.
"""

from __future__ import annotations

from retailiq.tools.base import Tool, ToolResult
from retailiq.tools.registry import ToolRegistry, default_registry, resolve_web_search
from retailiq.tools.tavily_search import TavilySearchTool
from retailiq.tools.web_search import WebSearchTool

__all__ = [
    "TavilySearchTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "WebSearchTool",
    "default_registry",
    "resolve_web_search",
]
