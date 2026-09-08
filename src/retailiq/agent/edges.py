"""Conditional edges — the agent's control flow.

Isolated from `nodes.py` so the branching rules can be unit-tested with
plain dicts and zero LLM calls. These functions are pure: state in, next-node
name out.
"""

from __future__ import annotations

from retailiq.agent.nodes import grade_generation
from retailiq.agent.state import AgentState
from retailiq.core.enums import GenerationVerdict, Route
from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings

logger = get_logger(__name__)

# Edge target labels. Named constants because the same strings appear in the
# graph's routing maps, and a typo there fails at runtime, not import time.
RETRIEVE = "retrieve"
WEB_SEARCH = "web_search"
DIRECT_ANSWER = "direct_answer"
GENERATE = "generate"
REWRITE = "rewrite"
REGENERATE = "regenerate"
DONE = "done"


def route_decision(state: AgentState) -> str:
    """Send the question to retrieval, the web, or a direct answer."""
    route = state.get("route", Route.VECTORSTORE)
    return {
        Route.VECTORSTORE: RETRIEVE,
        Route.WEB_SEARCH: WEB_SEARCH,
        Route.DIRECT_ANSWER: DIRECT_ANSWER,
    }[Route(route)]


def documents_decision(state: AgentState, settings: Settings | None = None) -> str:
    """After grading: generate if anything survived, else rewrite if in budget.

    When the rewrite budget is exhausted and nothing relevant was found, we
    still go to `generate` rather than dead-ending. The generator is
    instructed to say it doesn't have the information — an honest "not in the
    knowledge base" is a useful answer, and an error page is not.
    """
    settings = settings or get_settings()

    if state.get("documents"):
        return GENERATE

    if state.get("rewrites", 0) < settings.agent.max_query_rewrites:
        return REWRITE

    logger.info("Rewrite budget exhausted with no relevant documents; answering honestly")
    return GENERATE


def generation_decision(state: AgentState, settings: Settings | None = None) -> str:
    """After generation: finish, regenerate, or rewrite and retrieve again.

    Both corrective paths are budget-capped. A question the knowledge base
    genuinely cannot answer would otherwise cycle forever, burning tokens.
    """
    settings = settings or get_settings()
    verdict = grade_generation(state)

    if verdict is GenerationVerdict.USEFUL:
        return DONE

    if verdict is GenerationVerdict.NOT_GROUNDED:
        # Same context, another attempt — the context was fine, the draft wasn't.
        # `generation_attempts` counts the initial generation as 1, so this
        # permits exactly `max_generation_retries` further attempts.
        if state.get("generation_attempts", 0) <= settings.agent.max_generation_retries:
            return REGENERATE
        logger.warning("Generation retry budget exhausted; returning the last draft")
        return DONE

    # NOT_USEFUL: the context itself was inadequate, so change the query.
    if state.get("rewrites", 0) < settings.agent.max_query_rewrites:
        return REWRITE
    return DONE
