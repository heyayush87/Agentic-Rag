"""The agentic controller: a LangGraph state machine with self-correction.

Module split mirrors the two halves of any graph:

* `nodes.py` — what happens at each step (the work)
* `edges.py` — which step runs next (the control flow)

Keeping the branching predicates out of the node functions means the control
logic is unit-testable with plain dictionaries and no LLM calls at all.
"""

from __future__ import annotations

from retailiq.agent.graph import build_graph, get_compiled_graph
from retailiq.agent.state import AgentState

__all__ = ["AgentState", "build_graph", "get_compiled_graph"]
