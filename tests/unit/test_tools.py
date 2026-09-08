"""Tool contract and registry behaviour."""

from __future__ import annotations

import pytest

from retailiq.core.settings import Settings
from retailiq.tools.base import Tool, ToolResult
from retailiq.tools.registry import ToolRegistry, default_registry
from retailiq.tools.web_search import WebSearchTool

pytestmark = pytest.mark.unit


class _StubTool(Tool):
    name = "stub"
    description = "test double"

    def run(self, query: str, **kwargs: object) -> ToolResult:
        return ToolResult(content=f"ran: {query}", tool_name=self.name)


def test_registry_registers_and_runs() -> None:
    registry = ToolRegistry()
    registry.register(_StubTool())

    assert "stub" in registry
    assert registry.run("stub", "hello").content == "ran: hello"


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(_StubTool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_StubTool())


def test_unknown_tool_returns_a_failure_not_an_exception() -> None:
    """A missing tool must degrade the answer, not abort the graph."""
    result = ToolRegistry().run("does_not_exist", "q")
    assert result.success is False
    assert "not registered" in result.content


def test_web_search_disabled_by_configuration() -> None:
    settings = Settings(tools={"enable_web_search": False})  # type: ignore[arg-type]
    tool = WebSearchTool(settings)

    assert tool.is_available() is False
    result = tool.run("anything")
    assert result.success is False
    assert "disabled" in result.content


def test_web_search_failure_is_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline or rate-limited search must not take the agent down."""
    settings = Settings(tools={"enable_web_search": True})  # type: ignore[arg-type]

    from retailiq.tools import web_search as module

    monkeypatch.setattr(module, "_load_ddgs_client", lambda: None)

    result = WebSearchTool(settings).run("q")
    assert result.success is False
    assert "not installed" in result.content


def test_the_maintained_ddgs_package_is_preferred() -> None:
    """`duckduckgo_search` still imports but returns zero results without
    raising — a silent failure indistinguishable from "no answer exists".
    The loader must reach for `ddgs` first."""
    from retailiq.tools.web_search import _load_ddgs_client

    client = _load_ddgs_client()
    assert client is not None
    assert client.__module__.startswith("ddgs"), (
        f"expected the ddgs package, got {client.__module__}"
    )


def test_search_errors_are_reported_as_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A network error degrades the answer rather than crashing the graph."""
    settings = Settings(tools={"enable_web_search": True})  # type: ignore[arg-type]

    class _Exploding:
        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def text(self, *a: object, **k: object):
            raise ConnectionError("network unreachable")

    from retailiq.tools import web_search as module

    monkeypatch.setattr(module, "_load_ddgs_client", lambda: _Exploding)

    result = WebSearchTool(settings).run("q")
    assert result.success is False
    assert "ConnectionError" in result.content


def test_default_registry_contains_web_search() -> None:
    assert "web_search" in default_registry().names()
