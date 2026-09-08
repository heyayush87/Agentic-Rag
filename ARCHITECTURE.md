# Architecture & Design Notes

Why RetailIQ is built the way it is.

---

## 1. The problem with naive RAG

A naive pipeline is: embed the question → fetch top-k chunks → stuff them into
the prompt → generate. It fails in three predictable ways:

1. **Irrelevant retrieval.** Vector similarity is a *proxy* for relevance, not
   relevance itself. A question about *returning* a TV happily surfaces a chunk
   about *delivering* one. It still gets pasted into the prompt.
2. **No recovery.** If the first phrasing retrieves badly, there is no second
   chance.
3. **Hallucination.** The model produces fluent claims unsupported by the
   retrieved text, and nothing catches it.

In a retail setting, failure 3 is the dangerous one. A colleague told "you have
90 days to return that" when the policy says 30 has been actively misinformed by
a system that sounded authoritative.

## 2. The agentic solution

RetailIQ treats answering as a **control-flow problem** handled by a LangGraph
state machine, not a single forward pass.

| Capability | Node | Idea from |
|---|---|---|
| Route by question type | `route_question` | Adaptive-RAG |
| Grade each retrieved chunk | `grade_documents` | Self-RAG / CRAG |
| Rewrite and retry on weak retrieval | `rewrite_query` | CRAG |
| Grounded generation with citations | `generate` | — |
| Hallucination + usefulness gate | `grade_generation` | Self-RAG |
| LLM as controller over tools | the whole graph | Agentic RAG |

### 2.1 Routing (`route_question`)
Decides between the internal knowledge base, a web search, or a direct
conversational answer. Avoids spending retrieval on "hi", and lets the system
reach outside its documents when a question genuinely needs it. An unparseable
route degrades to `vectorstore` rather than crashing.

### 2.2 Relevance grading (`grade_documents`)
Every retrieved chunk is graded yes/no and the failures are discarded *before*
generation. This is the heart of self-RAG: the agent does not trust its own
retriever. It costs one cheap single-token call per chunk.

### 2.3 Corrective rewriting (`rewrite_query`)
If grading leaves nothing, the question is reformulated into a clearer,
keyword-rich form and retrieval runs again — up to `MAX_QUERY_REWRITES`.

Rewrites always derive from `original_question`, never from the previous
rewrite. Chaining rewrites compounds drift until the agent is answering a
question nobody asked.

### 2.4 Grounded generation (`generate`)
Answers come only from surviving context, with instructions to cite source
documents and to admit absence rather than guess.

### 2.5 The quality gate (`grade_generation`)
Two judges run before an answer is allowed out:

- **Grounded?** Is every claim supported by the context? If not → **regenerate**
  from the same context. The context was fine; the draft wasn't.
- **Useful?** Does it actually resolve the question? Grounded-but-evasive is
  still a failure. If not → **rewrite the query and retrieve again**. The
  context itself was inadequate.

The two failure modes need different remedies, which is exactly why they are
graded separately.

## 3. Bounded loops

Every corrective path is budgeted, because self-correction that cannot terminate
is a production incident:

| Budget | Default | Guards against |
|---|---|---|
| `MAX_QUERY_REWRITES` | 2 | Rewrite → retrieve → grade → rewrite forever |
| `MAX_GENERATION_RETRIES` | 2 | Ungrounded → regenerate forever |
| `RECURSION_LIMIT` | 25 | Any unforeseen cycle in the graph |

When the rewrite budget runs out with nothing relevant, the agent still goes to
`generate` rather than dead-ending. An honest "that isn't in the knowledge base"
is a useful answer; an error page is not.

## 4. Layering

```
api / cli / ui        transports — no business logic
      ↓
services              use cases; the only callers of the graph
      ↓
agent / ingestion     domain logic
      ↓
llm / tools           external capability adapters
      ↓
core                  settings, logging, errors — imports nothing above it
```

Two rules make this hold:

- **No transport touches the graph directly.** All three call `RAGService`, so
  an answer means the same thing over HTTP as in the terminal.
- **Nothing imports a provider SDK except `llm/factory.py`.** Swapping OpenAI
  for Groq is a one-line `.env` change; no business logic names a vendor.

`nodes.py` (what happens) is split from `edges.py` (what happens next) so the
control flow is unit-testable with plain dicts and zero LLM calls. That split is
what makes the highest-risk logic in the system — the retry budgets — cheap to
verify.

## 5. Key decisions

**Chunk size 800 / overlap 120, split on markdown headings first.**
The knowledge base is structured markdown where each heading is a self-contained
policy. Splitting on `\n## ` first means a chunk tends to be one whole rule
rather than half of two adjacent rules. A chunk that splits mid-policy is the
single most common cause of a confidently wrong RAG answer.

**Chat provider and embedding provider are configured independently.**
They are genuinely independent choices. Groq serves chat faster than anyone but
exposes *no* embedding endpoint, so the natural pairing is Groq chat +
on-device sentence-transformers. One combined setting could not express that.

**Embeddings must match between index and query time.**
Different models produce vectors in incompatible spaces. Changing
`EMBED_PROVIDER` requires a full re-ingest — searching a MiniLM index with
OpenAI vectors returns noise, silently and without error.

**`--reset` defaults to true on ingest.**
Chroma *appends* rather than replaces. Re-running without a reset duplicates
every chunk, which degrades retrieval by filling top-k with copies of the same
passage.

**Temperature 0 everywhere.**
Graders must not wander between runs, or the same question yields a different
verdict each time and the evaluation numbers stop meaning anything.

**Single-token grader outputs.**
Constraining graders to "yes"/"no" makes them parseable without JSON-mode
support (not every provider offers it) and nearly free, which matters because
grading runs once per retrieved chunk.

**Typed exceptions carry HTTP status.**
A missing index is `503`, not `500` — the service is configured correctly and
merely not ready. That distinction lets an orchestrator hold traffic back
instead of routing users into errors.

**Tools return results, never raise.**
A dead external API should degrade the answer, not crash the agent mid-graph.

## 6. Observability

Two layers, catching different failure classes:

| | pytest (pre-merge) | LangSmith (post-deploy) |
|---|---|---|
| Catches | Code defects: config wiring, control flow, retry budgets | Quality drift: hallucination rate, latency, cost |
| Determinism | Fully deterministic, offline, free | Live traffic, paid SaaS |
| On failure | Blocks the merge | Reports what already happened |

Neither substitutes for the other. A `KeyError` from an eagerly-evaluated
`dict.get` default is invisible to LangSmith until a user hits it; a prompt
change that quietly raises hallucination rate is invisible to pytest forever.

Structured logging ties them together: every request carries a `correlation_id`
(honoured from an inbound `X-Correlation-ID` header, so traces span services),
and `LOG_FORMAT=json` emits one object per line for a log aggregator.

## 7. Evaluation

Four metrics over a golden dataset, deliberately spanning both halves of the
pipeline so a regression can be localised:

| Metric | Answers |
|---|---|
| Retrieval hit-rate | Did the expected document reach the context? *(retrieval problem)* |
| Keyword recall | Did the answer state the required fact? *(deterministic, cheap)* |
| Faithfulness | Is every claim supported? *(offline analogue of the runtime gate)* |
| Answer relevance | Does it address the question? |

If hit-rate drops, the problem is chunking or embeddings. If hit-rate holds but
faithfulness drops, the problem is the prompt. One aggregate score would hide
that distinction.

Implemented without RAGAS on purpose: RAGAS pins its own provider stack, whereas
these judges run against whichever provider is configured — so the harness works
on a free Groq key or fully offline via Ollama.

## 8. Extending it

- **New tool:** subclass `Tool`, register it in `tools/registry.py`, add a route
  to the router prompt. No node changes.
- **New vector store:** `ingestion/vector_store.py` is the only file that names
  Chroma.
- **New paper/mechanism:** add it as a node and wire a conditional edge. The
  graph is designed to make that a small, local change.
