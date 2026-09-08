"""Graph nodes — the work performed at each step.

Every node takes the shared `AgentState` and returns a *partial* update that
LangGraph merges in. Nodes never decide what runs next; that lives in
`edges.py`.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from retailiq.agent import prompts
from retailiq.agent.state import AgentState, append_trace
from retailiq.core.enums import GenerationVerdict, Route
from retailiq.core.exceptions import RetrievalError, VectorStoreNotFoundError
from retailiq.core.logging import get_logger
from retailiq.core.settings import get_settings
from retailiq.llm.factory import get_chat_model
from retailiq.tools.registry import default_registry

logger = get_logger(__name__)

NO_CONTEXT_SENTINEL = "(no context retrieved)"
DIRECT_ANSWER_SENTINEL = "(no retrieval — direct answer)"


def _first_token(text: str) -> str:
    """Extract a grader's single-word verdict.

    Models append punctuation, quotes, or a stray explanatory sentence even
    when told not to. Taking the first token and stripping punctuation is
    more robust than exact matching against the whole response.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.split()[0].strip("\"'.,:;*`").lower()


def _user_question(state: AgentState) -> str:
    """The question as the user asked it, falling back to the current one.

    Written as an explicit `or` chain rather than
    ``state.get("original_question", state["question"])`` — Python evaluates
    a `.get` default *eagerly*, so that form raises `KeyError` whenever
    ``question`` is absent, even when ``original_question`` is present.
    """
    return state.get("original_question") or state.get("question") or ""


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_question(state: AgentState) -> AgentState:
    """Decide whether to retrieve internally, search the web, or answer directly."""
    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(content=prompts.ROUTER_SYSTEM),
            HumanMessage(content=state["question"]),
        ]
    )
    token = _first_token(str(response.content))

    try:
        route = Route(token)
    except ValueError:
        # An unparseable route must not abort the run. Internal retrieval is
        # the safe default for a retail knowledge assistant: worst case the
        # relevance grader discards the chunks and we fall through cleanly.
        logger.warning("Router returned an unknown route", extra={"raw": token})
        route = Route.VECTORSTORE

    return AgentState(route=route, trace=append_trace(state, f"router → {route}"))


# ---------------------------------------------------------------------------
# Retrieval and grading
# ---------------------------------------------------------------------------
def retrieve(state: AgentState) -> AgentState:
    """Fetch the top-k chunks for the current (possibly rewritten) question."""
    from retailiq.ingestion.vector_store import get_retriever

    try:
        documents = get_retriever().invoke(state["question"])
    except VectorStoreNotFoundError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Vector search failed: {exc}") from exc

    return AgentState(
        documents=documents,
        trace=append_trace(state, f"retrieved {len(documents)} chunks"),
    )


def grade_documents(state: AgentState) -> AgentState:
    """Discard retrieved chunks the model judges irrelevant.

    This is the heart of self-RAG: the agent does not trust its retriever.
    Vector similarity is a proxy for relevance, not relevance itself — a
    question about *returning* a TV will happily surface a chunk about
    *delivering* one. Feeding that to the generator invites a confident,
    wrong answer, so each chunk is judged before it is allowed to matter.
    """
    llm = get_chat_model()
    documents = state.get("documents", [])
    kept: list[Document] = []

    for document in documents:
        response = llm.invoke(
            [
                SystemMessage(content=prompts.GRADE_DOCUMENTS_SYSTEM),
                HumanMessage(
                    content=(f"Question: {state['question']}\n\nDocument:\n{document.page_content}")
                ),
            ]
        )
        if _first_token(str(response.content)) == "yes":
            kept.append(document)

    message = f"relevance grader kept {len(kept)}/{len(documents)} chunks"
    logger.info("Graded documents", extra={"kept": len(kept), "total": len(documents)})
    return AgentState(documents=kept, trace=append_trace(state, message))


def rewrite_query(state: AgentState) -> AgentState:
    """Reformulate the question after weak retrieval, then retry.

    Always rewrites from `original_question`, never from the previous
    rewrite — otherwise successive attempts drift further from what the user
    actually asked.
    """
    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(content=prompts.REWRITE_SYSTEM),
            HumanMessage(content=_user_question(state)),
        ]
    )
    rewritten = str(response.content).strip()

    return AgentState(
        question=rewritten,
        rewrites=state.get("rewrites", 0) + 1,
        trace=append_trace(state, f"rewrote query → {rewritten!r}"),
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def web_search(state: AgentState) -> AgentState:
    """Fall back to public web search and wrap the result as a document."""
    result = default_registry().run("web_search", state["question"])
    document = Document(
        page_content=result.content,
        metadata={"source": "web_search", "topic": "web", "success": result.success},
    )
    return AgentState(
        documents=[document],
        trace=append_trace(state, f"used web_search tool (success={result.success})"),
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(state: AgentState) -> AgentState:
    """Compose a grounded answer from whatever context survived grading."""
    documents = state.get("documents", [])
    context = (
        "\n\n---\n\n".join(
            f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in documents
        )
        or NO_CONTEXT_SENTINEL
    )

    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(content=prompts.GENERATE_SYSTEM),
            HumanMessage(content=(f"Context:\n{context}\n\nQuestion: {_user_question(state)}")),
        ]
    )

    attempts = state.get("generation_attempts", 0) + 1
    return AgentState(
        context=context,
        generation=str(response.content).strip(),
        generation_attempts=attempts,
        trace=append_trace(
            state,
            "generated answer" if attempts == 1 else f"regenerated answer (attempt {attempts})",
        ),
    )


def direct_answer(state: AgentState) -> AgentState:
    """Answer conversational questions with no retrieval at all."""
    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(content=prompts.DIRECT_ANSWER_SYSTEM),
            HumanMessage(content=state["question"]),
        ]
    )
    return AgentState(
        generation=str(response.content).strip(),
        context=DIRECT_ANSWER_SENTINEL,
        trace=append_trace(state, "answered directly"),
    )


# ---------------------------------------------------------------------------
# Post-generation quality gate
# ---------------------------------------------------------------------------
def grade_generation(state: AgentState) -> GenerationVerdict:
    """Two-stage check before an answer is allowed out.

    1. **Grounded?** Is every claim supported by the retrieved context? This
       is the hallucination gate — the guardrail that makes the system safe
       to point at a store colleague.
    2. **Useful?** Grounded but evasive ("I don't have that information") is
       still a failure from the user's point of view, so a second judge asks
       whether the question was actually resolved.

    The two failures need different remedies, which is why they are separate:
    ungrounded means *regenerate from the same context*; not useful means the
    context was wrong, so *rewrite the query and retrieve again*.
    """
    context = state.get("context", "")

    # Direct answers and empty-context runs have nothing to check against;
    # grading them would compare an answer to the string "(no context...)".
    if not context or context.startswith("(no"):
        return GenerationVerdict.USEFUL

    llm = get_chat_model()

    grounded = _first_token(
        str(
            llm.invoke(
                [
                    SystemMessage(content=prompts.HALLUCINATION_SYSTEM),
                    HumanMessage(
                        content=f"Context:\n{context}\n\nAnswer:\n{state.get('generation', '')}"
                    ),
                ]
            ).content
        )
    )
    if grounded != "yes":
        logger.warning("Answer failed the grounding check")
        return GenerationVerdict.NOT_GROUNDED

    useful = _first_token(
        str(
            llm.invoke(
                [
                    SystemMessage(content=prompts.ANSWER_QUALITY_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Question: {_user_question(state)}\n\n"
                            f"Answer:\n{state.get('generation', '')}"
                        )
                    ),
                ]
            ).content
        )
    )
    return GenerationVerdict.USEFUL if useful == "yes" else GenerationVerdict.NOT_USEFUL


def settings_snapshot() -> dict[str, object]:
    """Configuration echoed into logs at graph construction time."""
    return get_settings().describe()
