.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-dev ingest ask api ui eval test test-unit test-integration lint format typecheck check clean docker-build docker-up docker-down

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the default runnable stack (Groq chat + local embeddings)
	$(PY) -m pip install -e ".[all]"

install-dev: ## Install runtime + development tooling
	$(PY) -m pip install -e ".[all,dev]"
	pre-commit install

ingest: ## Build (or rebuild) the vector index from data/knowledge_base
	retailiq ingest --reset

ask: ## Ask one question, e.g. make ask Q="how long to return an electrical item?"
	retailiq ask "$(Q)"

api: ## Serve the REST API with hot reload
	uvicorn retailiq.api.main:app --reload --host 0.0.0.0 --port 8000

ui: ## Launch the Streamlit demo UI
	streamlit run src/retailiq/ui/streamlit_app.py

eval: ## Run the evaluation harness
	retailiq evaluate

test: ## Full test suite with coverage
	pytest --cov --cov-report=term-missing

test-unit: ## Offline unit tests only
	pytest -m unit

test-integration: ## Integration tests
	pytest -m integration

lint: ## Lint with ruff
	ruff check src tests

format: ## Auto-format and fix imports
	ruff format src tests
	ruff check --fix src tests

typecheck: ## Static type check with mypy
	mypy

check: lint typecheck test-unit ## Everything CI runs on a pull request

clean: ## Remove caches, build output and the local vector store
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build var/chroma
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

docker-build: ## Build the container image
	docker build -f deploy/Dockerfile -t retailiq:local .

docker-up: ## Start the API + UI stack
	docker compose -f deploy/docker-compose.yml up --build

docker-down: ## Stop the stack
	docker compose -f deploy/docker-compose.yml down
