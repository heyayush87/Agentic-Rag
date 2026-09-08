# 🛒 Retail Agentic RAG Assistant

A **self-correcting, agentic Retrieval-Augmented Generation (RAG)** system that
automates internal knowledge lookup for retail operations — returns policy,
loyalty/Clubcard rules, store operations, supplier & logistics procedures, and
the product catalogue.

Unlike a basic "retrieve-then-answer" RAG, this system uses an **agent** (built
with **LangGraph**) that reasons about *how* to answer:

- **Routes** each question to the internal knowledge base, a web-search tool, or
  a direct answer.
- **Grades its own retrieved documents** for relevance and throws away the noise.
- **Rewrites the query and retries** when retrieval is weak.
- **Checks its answer for hallucination** and usefulness before responding, and
  loops back if the answer isn't grounded.
- Ships with an **evaluation harness** that measures retrieval hit-rate,
  faithfulness, and answer relevance — so the quality is a number, not a vibe.

> **Why this framing?** In a retail/automation context, the value of AI is
> *trustworthy automation of manual work* — here, the manual work of a colleague
> looking up a policy. The self-grading and hallucination checks are exactly the
> guardrails that make such automation safe to deploy.

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

A rendered version and a deeper walkthrough are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Quick start

### 1. Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Choose a provider (any one works)
Copy `.env.example` to `.env` and set `LLM_PROVIDER`:

| Provider | Cost | What to set |
|----------|------|-------------|
| **Groq** (recommended, free & fast) | Free tier | `LLM_PROVIDER=groq`, `GROQ_API_KEY=…`, `EMBED_PROVIDER=local` |
| **Google Gemini** | Free tier | `LLM_PROVIDER=google`, `GOOGLE_API_KEY=…`, `EMBED_PROVIDER=google` |
| **OpenAI** | Paid | `LLM_PROVIDER=openai`, `OPENAI_API_KEY=…`, `EMBED_PROVIDER=openai` |
| **Ollama** | Free, fully local, no key | `LLM_PROVIDER=ollama`, `EMBED_PROVIDER=ollama` (or `local`) |

`EMBED_PROVIDER=local` uses a small on-device sentence-transformers model, so
embeddings stay free regardless of which chat provider you pick.

### 3. Build the knowledge index
```bash
python -m src.ingest
```

### 4. Ask questions
```bash
python main.py "How long do I have to return an electrical item?"
python main.py                       # interactive REPL
```

### 5. Run the demo UI (great for interviews)
```bash
streamlit run app.py
```

### 6. Evaluate
```bash
python evaluate.py
```

---

## Project layout
```
retail-agentic-rag/
├── data/                     # the retail knowledge base + eval set
├── src/
│   ├── llm.py                # provider-flexible model/embedding factory
│   ├── ingest.py             # load → chunk → embed into Chroma
│   ├── prompts.py            # every reasoning prompt in one readable place
│   ├── tools.py              # web-search fallback tool
│   ├── nodes.py              # router, retrieve, grade, rewrite, generate, graders
│   └── graph.py              # LangGraph state machine (the agent controller)
├── main.py                   # CLI + REPL
├── app.py                    # Streamlit demo UI
├── evaluate.py               # RAGAS-style metrics
└── tests/                    # import/structure smoke test
```

## Tech
Python · LangGraph · LangChain · Chroma (vector store) · sentence-transformers ·
Streamlit · pluggable LLMs (Groq / Gemini / OpenAI / Ollama).

## Extending it
- Swap the `data/` docs for real store documentation (PDFs supported via a loader change).
- Add tools to `src/tools.py` (e.g. a live stock-lookup API) and a new route in the router.
- Point `EMBED_PROVIDER`/`LLM_PROVIDER` at a hosted model for production.

See [`RESUME.md`](RESUME.md) for ready-to-paste bullet points and interview talking points.
