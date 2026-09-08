# Resume bullets & interview talking points

Fill in the `X%` from your own `python evaluate.py` run.

## Resume bullet options (pick 1–2)

**Concise:**
> Built a self-correcting **agentic RAG** assistant (Python, LangGraph, Chroma)
> that automates retail knowledge lookup — with LLM-based query routing,
> retrieved-document relevance grading, corrective query rewriting, and
> hallucination checks — achieving **X% faithfulness** and **X% answer
> relevance** on a labelled evaluation set.

**Impact-oriented (good for a retail/automation role):**
> Designed and built an agentic RAG system to automate first-line retail
> operations support (returns, loyalty, store SOPs, supply chain), replacing
> manual policy lookups. Implemented a LangGraph controller with self-grading
> and query-rewrite loops to keep answers grounded, and a RAGAS-style evaluation
> harness (retrieval hit-rate, faithfulness, answer relevance).

**Engineering-oriented:**
> Engineered a provider-agnostic (OpenAI/Groq/Gemini/Ollama) agentic RAG
> pipeline with a Chroma vector store, bounded self-correction loops, graceful
> tool fallback, and an automated evaluation suite; shipped a Streamlit demo
> exposing the agent's decision trace.

## One-line resume skill tags
`Agentic RAG · LangGraph · LangChain · Vector Databases (Chroma) · LLM Evaluation (RAGAS-style) · Python · Streamlit`

---

## Interview talking points

**"Walk me through the project."**
> It's a RAG assistant for retail operations, but instead of a single
> retrieve-then-answer pass, an agent controls the flow. It routes the question,
> grades whether the retrieved documents are actually relevant, rewrites and
> retries the query if they aren't, and checks its own answer for hallucination
> before returning it. I also built an evaluation harness so I can quantify
> faithfulness and answer relevance rather than eyeballing it.

**"How is this different from a basic RAG tutorial?"**
> Three things: relevance grading of retrieved chunks (CRAG/Self-RAG),
> corrective query rewriting with a bounded retry loop, and a self-check on the
> generated answer. Plus measured quality via an eval set.

**"Why is this relevant to retail / automation?"**
> The whole point is *trustworthy automation of manual work* — a colleague no
> longer has to hunt through policy documents. The self-grading and
> hallucination guardrails are what make that automation safe to actually
> deploy, which is the hard part in a regulated, customer-facing environment.

**"What would you improve for production?"**
> Add tracing/observability (e.g. LangSmith), a caching layer, real document
> ingestion (PDFs, Confluence), a reranker before grading to cut LLM calls,
> access control on sources, and an offline evaluation pipeline in CI so I catch
> regressions when prompts or models change.

**"What was the hardest part?"**
> Designing the control flow so the self-correction loop always terminates —
> bounding rewrites and setting a graph recursion limit — while still recovering
> from bad retrieval.
