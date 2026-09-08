"""External tools available to the agent. Currently: web search fallback.

Designed to degrade gracefully — if the search package or network isn't
available, it returns a clear stub instead of crashing the graph, so the
project still runs end-to-end in a locked-down/offline environment.
"""
from __future__ import annotations

import config


def web_search(query: str, max_results: int = 4) -> str:
    """Return a concatenated snippet block from a web search, or a stub."""
    if not config.ENABLE_WEB_SEARCH:
        return "[web search disabled by configuration]"
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                results.append(f"{title}\n{body}\n(source: {href})")
        if results:
            return "\n\n".join(results)
        return "[no web results found]"
    except Exception as exc:  # pragma: no cover - network/dependency dependent
        return (
            "[web search unavailable: "
            f"{type(exc).__name__}. Install duckduckgo-search and ensure network "
            "access, or set ENABLE_WEB_SEARCH=false.]"
        )
