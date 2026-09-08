# Architecture & Design Notes

This document explains *why* the system is built the way it is — the part that
matters in an interview.

## The problem with "naive" RAG
A naive RAG pipeline is: embed the question → fetch top-k chunks → stuff them
into the prompt → generate. It fails in three common ways:

1. **Irrelevant retrieval** — the top-k chunks may not actually answer the
   question, but they still get pasted into the prompt and pollute the answer.
2. **No recovery** — if the first query is phrased poorly, there's no second
   chance.
3. **Hallucination** — the model can produce fluent claims not supported by the
   retrieved text, and nothing catches it.

## The agentic solution
This project treats answering as a **control-flow problem** handled by an agent
(a LangGraph state machine), not a single forward pass. Each capability below
maps to a node in `src/nodes.py`.

### 1. Query routing (`route_question`)
An LLM decides whether a question belongs to the internal knowledge base, needs
a web search, or is just conversational. This avoids wasting retrieval on "hi"
and lets the system reach outside its documents when appropriate. *Inspired by
adaptive-RAG routing.*

### 2. Document relevance grading (`grade_documents`)
Every retrieved chunk is graded yes/no for relevance and irrelevant chunks are
discarded before generation. This is the core idea of **Self-RAG / CRAG**
(Corrective RAG): don't trust the retriever blindly.

### 3. Corrective query rewriting (`rewrite_query`)
If grading leaves no relevant chunks, the agent rewrites the question into a
clearer, keyword-rich form and retrieves again — up to `MAX_QUERY_REWRITES`
times. This is the "corrective" loop of CRAG.

### 4. Grounded generation (`generate`)
The answer is generated from only the surviving context, with an instruction to
cite source documents and to admit when the information isn't present.

### 5. Self-checking (`grade_generation`)
Two graders run on the output:
- **Faithfulness / hallucination**: is every claim supported by the context? If
  not → regenerate.
- **Usefulness**: does the answer actually address the question? If not →
  rewrite the query and retry (if budget remains).

This closes the loop: the agent won't return an answer it judges ungrounded.

## State
All nodes share a typed `GraphState` (`src/nodes.py`) carrying the question,
retrieved documents, context, generation, a rewrite counter (loop budget), and a
human-readable `trace` of every decision — which the UI surfaces so the agentic
behaviour is visible.

## Design choices worth defending in an interview
- **Provider-flexibility** (`src/llm.py`): the codebase depends on no single LLM
  vendor. Good engineering hygiene and lets it run free (Groq + local
  embeddings) or fully offline (Ollama).
- **Local embeddings by default**: no per-query embedding cost, and the vector
  store is fully reproducible.
- **Bounded loops**: `MAX_QUERY_REWRITES` and a graph `recursion_limit` prevent
  infinite self-correction — a real production concern.
- **Graceful tool degradation** (`src/tools.py`): the web-search tool returns a
  clear stub instead of crashing when offline, so the pipeline always completes.
- **Measured, not asserted**: `evaluate.py` reports retrieval hit-rate,
  faithfulness, and answer relevance on a labelled set.

## How this maps to the research literature
| Paper / idea | Where it lives here |
|--------------|--------------------|
| Self-RAG (self-reflection, relevance & support tokens) | `grade_documents`, `grade_generation` |
| CRAG — Corrective RAG (evaluate retrieval, correct via rewrite/search) | `grade_documents` → `rewrite`/`web_search` |
| Adaptive-RAG (route by question complexity/type) | `route_question` |
| Agentic RAG (LLM as controller over tools & retrieval) | the whole `src/graph.py` |

If you were handed a specific paper, the cleanest way to slot it in is to add
its mechanism as a node and wire a conditional edge — the graph is designed to
make that a small, local change.
