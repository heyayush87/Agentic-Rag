# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-09-08

Restructured from a flat demo layout into a layered, installable platform.

### Added
- **Installable package** `retailiq` under a `src/` layout with `pyproject.toml`,
  a `retailiq` console script, and provider SDKs split into optional extras so a
  deployment installs only what it uses.
- **Typed configuration** (`core/settings.py`) via pydantic-settings — validated
  at startup, secrets wrapped in `SecretStr`, composed from per-concern sections.
- **Structured logging** (`core/logging.py`) with correlation IDs propagated
  through a `contextvar`, plus a JSON formatter for log aggregators.
- **Typed exception hierarchy** (`core/exceptions.py`) carrying HTTP status codes
  and stable machine-readable error codes.
- **Domain models** (`domain/models.py`) — transport-agnostic `QueryResult`,
  `RetrievedChunk`, `EvaluationReport`.
- **Service layer** (`services/`) as the single entry point for every transport.
- **REST API** (`api/`) on FastAPI: `/api/v1/query`, `/health`, `/admin/ingest`,
  `/admin/evaluate`, with correlation-ID middleware and error mapping.
- **CLI** (`cli/`) on Typer: `ingest`, `ask`, `chat`, `evaluate`, `config`, `serve`.
- **Tool registry** (`tools/`) with a `Tool` contract that returns results rather
  than raising, so a failing tool degrades the answer instead of the run.
- **LangSmith tracing** (`core/observability.py`), optional and off by default.
- **Test suite** — 79 offline tests across `unit` / `integration` / `e2e` markers.
- **Docker** multi-stage build running as a non-root user, plus compose stack.
- **CI** — lint, type check, tests on Python 3.10–3.12, a tracked-`.env` guard,
  dependency audit, and an image build.
- `CONTRIBUTING.md`, `LICENSE`, and an expanded `ARCHITECTURE.md`.

### Changed
- `agent/nodes.py` split from `agent/edges.py`, separating work from control
  flow so retry budgets are unit-testable without an LLM.
- Ingestion split into `loaders` / `chunking` / `vector_store` / `pipeline`.
- Knowledge base moved to `data/knowledge_base/`; eval set to `data/eval/`.
- Vector store now persists to `var/chroma/` (git-ignored) rather than the repo.
- Generation retries are now explicitly budgeted by `MAX_GENERATION_RETRIES`;
  previously the ungrounded → regenerate edge had no counter and could cycle.

### Fixed
- Config values passed by field name were silently discarded, because
  `validation_alias` makes the alias the only accepted key without
  `populate_by_name`. Every runtime override was being ignored.
- `state.get("original_question", state["question"])` raised `KeyError` when
  `question` was absent, since Python evaluates a `.get` default eagerly — even
  when the requested key is present.
- `API_CORS_ORIGINS=a.com,b.com` crashed at startup: pydantic-settings
  JSON-decodes complex types from the environment before validators run.
  Resolved with `NoDecode`.
- Ingestion crashed when `DATA_DIR` pointed outside the project root, because
  `Path.relative_to` raises rather than returning an absolute path — which is
  the normal case for a container bind-mount.
- `ingest --reset` failed on Windows with `PermissionError: [WinError 32]` once
  the index had been queried, as `shutil.rmtree` cannot unlink memory-mapped
  files. Now drops the collection through Chroma's API, with directory removal
  as a fallback.

### Removed
- Flat-layout modules `config.py`, `main.py`, `app.py`, `evaluate.py` and
  `src/*.py`, superseded by the package.
- `requirements.txt`, replaced by `pyproject.toml` dependencies and extras.
