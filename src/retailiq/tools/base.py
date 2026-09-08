"""Tool contract.

Every tool returns a `ToolResult` rather than raising on failure. A dead
external API should degrade the answer, not crash the agent mid-graph — the
generator can still respond honestly from whatever context it does have.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a tool invocation."""

    content: str
    success: bool = True
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, tool_name: str, reason: str) -> ToolResult:
        """A non-fatal failure the agent can carry forward as context."""
        return cls(
            content=f"[{tool_name} unavailable: {reason}]",
            success=False,
            tool_name=tool_name,
            metadata={"reason": reason},
        )


class Tool(ABC):
    """Base class for an agent-callable tool."""

    #: Stable identifier used in routing decisions and traces.
    name: str = "tool"
    #: Shown to the router LLM to help it decide when this tool applies.
    description: str = ""

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> ToolResult:
        """Execute the tool. Must not raise — return `ToolResult.failure`."""

    def is_available(self) -> bool:
        """Whether the tool can run right now (deps installed, enabled)."""
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} available={self.is_available()}>"
