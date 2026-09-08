"""The individual reasoning steps ("nodes") of the agentic RAG graph.

Each node takes the shared `GraphState` and returns a partial update. The graph
(src/graph.py) wires them together with conditional edges that create the
self-correcting loop:

    route -> retrieve -> grade docs -> (rewrite -> retrieve)?  -> generate
                                    \-> web search -> generate
    generate -> check hallucination + answer quality -> (retry / rewrite / done)
"""
from __future__ import annotations

from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

import config
from src import prompts
from src.llm import get_chat_model
from src.tools import web_search


# --------------------------------------------------------------------------
# Shared state passed between every node
# --------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    question: str          # the (possibly rewritten) question used for retrieval
    original_question: str # what the user actually asked
    route: str             # vectorstore | web_search | answer
    documents: List[Document]
    context: str
    generation: str
    rewrites: int          # how many times we've rewritten the query
    trace: List[str]       # human-readable log of decisions, for the UI/README


def _log(state: GraphState, msg: str) -> List[str]:
    trace = list(state.get("trace", []))
    trace.append(msg)
    return trace


def _one_word(text: str) -> str:
    return text.strip().split()[0].strip('."\'').lower() if text.strip() else ""


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def route_question(state: GraphState) -> GraphState:
    """Agentic decision: retrieve internally, search the web, or answer directly."""
    llm = get_chat_model()
    resp = llm.invoke([
        SystemMessage(content=prompts.ROUTER_SYSTEM),
        HumanMessage(content=state["question"]),
    ])
    route = _one_word(resp.content)
    if route not in {"vectorstore", "web_search", "answer"}:
        route = "vectorstore"  # safe default for a retail KB assistant
    return {"route": route, "trace": _log(state, f"router → {route}")}


def retrieve(state: GraphState) -> GraphState:
    """Pull the top-k chunks from the vector store for the current question."""
    from src.ingest import get_retriever  # lazy import so import-time has no side effects

    docs = get_retriever().invoke(state["question"])
    return {
        "documents": docs,
        "trace": _log(state, f"retrieved {len(docs)} chunks"),
    }


def grade_documents(state: GraphState) -> GraphState:
    """Keep only chunks the LLM judges relevant. This is the core of self-RAG:
    the agent does not trust the retriever blindly."""
    llm = get_chat_model()
    kept: List[Document] = []
    for d in state.get("documents", []):
        resp = llm.invoke([
            SystemMessage(content=prompts.GRADE_DOCS_SYSTEM),
            HumanMessage(content=f"Question: {state['question']}\n\nDocument:\n{d.page_content}"),
        ])
        if _one_word(resp.content) == "yes":
            kept.append(d)
    msg = f"relevance grader kept {len(kept)}/{len(state.get('documents', []))} chunks"
    return {"documents": kept, "trace": _log(state, msg)}


def rewrite_query(state: GraphState) -> GraphState:
    """When retrieval was poor, rephrase the question and try again."""
    llm = get_chat_model()
    resp = llm.invoke([
        SystemMessage(content=prompts.REWRITE_SYSTEM),
        HumanMessage(content=state.get("original_question", state["question"])),
    ])
    new_q = resp.content.strip()
    return {
        "question": new_q,
        "rewrites": state.get("rewrites", 0) + 1,
        "trace": _log(state, f"rewrote query → {new_q!r}"),
    }


def do_web_search(state: GraphState) -> GraphState:
    """Fallback tool use: search the web and wrap results as a document."""
    results = web_search(state["question"])
    doc = Document(page_content=results, metadata={"source": "web_search"})
    return {
        "documents": [doc],
        "trace": _log(state, "used web_search tool"),
    }


def generate(state: GraphState) -> GraphState:
    """Compose a grounded answer from whatever context survived grading."""
    docs = state.get("documents", [])
    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    ) or "(no context retrieved)"

    llm = get_chat_model()
    resp = llm.invoke([
        SystemMessage(content=prompts.GENERATE_SYSTEM),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state.get('original_question', state['question'])}"),
    ])
    return {
        "context": context,
        "generation": resp.content.strip(),
        "trace": _log(state, "generated answer"),
    }


def direct_answer(state: GraphState) -> GraphState:
    """Answer conversational questions with no retrieval."""
    llm = get_chat_model()
    resp = llm.invoke([
        SystemMessage(content="You are a friendly retail operations assistant. Answer briefly."),
        HumanMessage(content=state["question"]),
    ])
    return {
        "generation": resp.content.strip(),
        "context": "(no retrieval — direct answer)",
        "trace": _log(state, "answered directly"),
    }


# --------------------------------------------------------------------------
# Graders used for conditional routing (return strings the graph branches on)
# --------------------------------------------------------------------------
def grade_generation(state: GraphState) -> str:
    """Check the generated answer for (a) grounding and (b) usefulness.

    Returns one of: "useful", "not_grounded", "not_useful". The graph uses this
    to decide whether to finish, regenerate, or rewrite-and-retry.
    """
    # If we have no context at all (e.g. web search stub, direct answer), accept.
    if state.get("context", "").startswith("(no"):
        return "useful"

    llm = get_chat_model()
    grounded = _one_word(llm.invoke([
        SystemMessage(content=prompts.HALLUCINATION_SYSTEM),
        HumanMessage(content=f"Context:\n{state.get('context','')}\n\nAnswer:\n{state.get('generation','')}"),
    ]).content)

    if grounded != "yes":
        return "not_grounded"

    useful = _one_word(llm.invoke([
        SystemMessage(content=prompts.ANSWER_QUALITY_SYSTEM),
        HumanMessage(content=f"Question: {state.get('original_question', state['question'])}\n\nAnswer:\n{state.get('generation','')}"),
    ]).content)

    return "useful" if useful == "yes" else "not_useful"
