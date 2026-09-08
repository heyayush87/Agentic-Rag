# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"
pre-commit install
cp .env.example .env      # then add one provider key
```

## Before you push

```bash
make check      # ruff + mypy + unit tests — the same gate CI runs
```

`pre-commit` runs formatting, linting, type checks and a private-key scan on
every commit. Don't bypass it with `--no-verify`.

## Architecture rules

The layering is enforced by review, not by a tool, so it needs stating:

```
api / cli / ui  →  services  →  agent / ingestion  →  llm / tools  →  core
```

1. **`core` imports nothing from feature packages.** The arrow points one way.
2. **No transport calls the graph directly.** `api`, `cli` and `ui` all go
   through `services`, so behaviour can't diverge between them.
3. **Only `llm/factory.py` imports a provider SDK.** Everything else calls
   `get_chat_model()` / `get_embeddings()`.
4. **Nodes do work; edges decide flow.** Keep branching predicates in
   `agent/edges.py` so control flow stays testable without an LLM.

## Tests

| Marker | Meaning | Needs |
|---|---|---|
| `unit` | Pure logic, fakes for models | Nothing |
| `integration` | Real components wired together | Nothing (fake embeddings) |
| `e2e` | Live provider and a built index | API key; auto-skipped otherwise |

```bash
pytest -m unit          # fast inner loop
pytest -m integration
pytest --cov            # full suite with coverage
```

**Unit and integration tests must never require an API key or network access.**
A suite that needs credentials is a suite nobody runs in CI.

When you fix a bug, add the test that would have caught it.

## Adding things

**A tool:** subclass `Tool` in `tools/`, register it in `registry.py`, and add a
route to the router prompt. Tools return `ToolResult`; they never raise.

**A provider:** add a branch to `_build_chat_model` / `_build_embeddings`, an
enum member, an extra in `pyproject.toml`, and the variables to `.env.example`.

**A config value:** add it to the relevant `*Settings` class with a
`validation_alias`, document it in `.env.example`, and add a validation test.

## Commits

Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
Never commit `.env` — CI fails the build if it is ever tracked.
