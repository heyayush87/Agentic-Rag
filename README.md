# 🛒 RetailIQ

**Enterprise agentic RAG platform for retail operations knowledge.**

A self-correcting retrieval system that automates internal knowledge lookup for
retail operations — returns policy, loyalty/Clubcard rules, store operations,
supplier and logistics procedures, and the product catalogue.

Unlike a basic "retrieve-then-answer" pipeline, RetailIQ uses an **agent**
(built with **LangGraph**) that reasons about *how* to answer:

- **Routes** each question to the internal knowledge base, a web-search tool, or a direct answer.
- **Grades its own retrieved documents** for relevance and discards the noise.
- **Rewrites the query and retries** when retrieval is weak.
- **Checks its answer for hallucination** and usefulness before responding, looping back if the answer isn't grounded.
- Ships with an **evaluation harness** so quality is a number, not a vibe.

> **Why this framing?** In retail automation, the value of AI is *trustworthy
> automation of manual work* — here, the manual work of a colleague looking up a
> policy. The self-grading and hallucination checks are exactly the guardrails
> that make such automation safe to deploy.

---

## Architecture

```
                              ┌─────────────┐
        question ───────────► │   ROUTER    │  (agentic decision)
                              └──────┬──────┘
             ┌───────────────┬───────┴───────────────┐
     "vectorstore"       "web_search"             "answer"
             ▼                ▼                       ▼
        ┌─────────┐     ┌───────────┐          ┌───────────────┐
        │RETRIEVE │     │ WEB SEARCH│          │ DIRECT ANSWER │──► END
        └────┬────┘     └─────┬─────┘          └───────────────┘
             ▼                │
     ┌───────────────┐        │
     │ GRADE DOCS    │        │
     │ (relevance)   │        │
     └──┬─────────┬──┘        │
   relevant   none & budget   │
        │      left → REWRITE │
        │          │          │
        ▼          └────► RETRIEVE (loop)
   ┌──────────┐               │
   │ GENERATE │ ◄─────────────┘
   └────┬─────┘
        ▼
 ┌─────────────────────┐
 │ GRADE GENERATION    │  hallucination + usefulness check
 └──┬──────────┬───────┘
 useful   not grounded → GENERATE (retry)
    │      not useful  → REWRITE → RETRIEVE (if budget)
    ▼
   END
```

A deeper walkthrough lives in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Project structure

```
retail-agentic-rag/
├── src/retailiq/
│   ├── core/              # settings, logging, exceptions, enums, observability
│   ├── domain/            # transport-agnostic result models
│   ├── llm/               # provider-flexible model & embedding factory
│   ├── ingestion/         # load → chunk → embed → persist
│   ├── agent/             # state, prompts, nodes, edges, LangGraph assembly
│   ├── tools/             # tool contract + registry (web search)
│   ├── services/          # use-case layer (RAG, evaluation)
│   ├── api/               # FastAPI: routers, schemas, middleware, deps
│   ├── cli/               # Typer CLI
│   └── ui/                # Streamlit demo
├── data/
│   ├── knowledge_base/    # the retail source documents
│   └── eval/              # golden question set
├── tests/{unit,integration,e2e}/
├── deploy/                # Dockerfile + docker-compose
├── .github/workflows/     # CI: lint, types, tests, secret scan, image build
└── pyproject.toml
```

**Layering rule:** dependencies point one way — `api`/`cli`/`ui` → `services` →
`agent`/`ingestion` → `llm`/`tools` → `core`. Nothing in `core` imports a
feature package, and no transport touches the graph directly.

---

## Quick start

### 1. Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

### 2. Choose a provider
Copy `.env.example` to `.env` and set `LLM_PROVIDER`:

| Provider | Cost | What to set |
|----------|------|-------------|
| **Groq** (recommended) | Free tier | `LLM_PROVIDER=groq`, `GROQ_API_KEY=…`, `EMBED_PROVIDER=local` |
| **Google Gemini** | Free tier | `LLM_PROVIDER=google`, `GOOGLE_API_KEY=…`, `EMBED_PROVIDER=google` |
| **OpenAI** | Paid | `LLM_PROVIDER=openai`, `OPENAI_API_KEY=…`, `EMBED_PROVIDER=openai` |
| **Ollama** | Free, fully local | `LLM_PROVIDER=ollama`, `EMBED_PROVIDER=ollama` |

`EMBED_PROVIDER=local` runs a small on-device sentence-transformers model, so
embeddings stay free regardless of which chat provider you pick.

### 3. Build the index and ask
```bash
retailiq ingest --reset
retailiq ask "How long do I have to return an electrical item?"
retailiq chat                 # interactive
retailiq config               # show effective settings (no secrets)
```

### 4. Run the interfaces
```bash
retailiq serve --reload                              # REST API → http://localhost:8000/docs
streamlit run src/retailiq/ui/streamlit_app.py       # demo UI
retailiq evaluate                                    # quality metrics
```

### 5. Docker
```bash
make docker-up      # API on :8000, UI on :8501
```

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/query` | Ask a question; returns answer, sources, grounding flag, trace |
| `GET`  | `/health` | Readiness — reports `degraded` when the index is missing |
| `POST` | `/admin/ingest` | Rebuild the vector index |
| `POST` | `/admin/evaluate` | Run the evaluation harness |

```bash
curl -X POST localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How long do I have to return an electrical item?"}'
```

Every response carries the route taken, the source documents, whether the answer
passed the grounding check, and how many rewrites it took — so a caller can
decide how much to trust it rather than taking the text on faith.

---

## Quality

```bash
make check          # lint + types + unit tests
make test           # full suite with coverage
```

- **79 offline tests** across unit and integration — no API key, no network, so CI runs them on every push.
- **Evaluation harness** scores retrieval hit-rate, keyword recall, faithfulness and answer relevance against a golden set.
- **LangSmith tracing** (optional) for production observability — set `LANGCHAIN_TRACING_V2=true` and a key.

Tests and tracing cover different failure classes: pytest catches *code* defects
before merge; LangSmith catches *quality* drift after deploy. Neither replaces
the other. Tests are excluded from the container image.

---

## Tech

Python 3.10+ · LangGraph · LangChain · Chroma · sentence-transformers · FastAPI ·
Typer · Streamlit · pydantic-settings · pytest · ruff · mypy · Docker ·
pluggable LLMs (Groq / Gemini / OpenAI / Ollama).

## Extending it

- Swap `data/knowledge_base/` for real store documentation — add a loader branch in `ingestion/loaders.py` for PDFs.
- Add a tool: subclass `Tool` in `tools/`, register it in `registry.py`, add a route in the router prompt.
- Migrate the vector store: `ingestion/vector_store.py` is the only file that knows about Chroma.
