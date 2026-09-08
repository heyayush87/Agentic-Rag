"""Typed, validated configuration.

Replaces module-level globals read at import time with `pydantic-settings`
models. Three things this buys us over `os.getenv` scattered through the code:

1. **Validation at startup.** A bad ``RETRIEVE_K=abc`` or an unknown provider
   fails immediately with a readable message, not hours later mid-request.
2. **Secrets stay wrapped.** API keys are `SecretStr`, so an accidental log
   line or repr prints ``**********`` instead of the key.
3. **Testability.** `get_settings` is cached; tests override it via the
   FastAPI dependency or by clearing the cache, with no env mutation.

Environment variable names are unchanged from the original flat config, so
existing `.env` files keep working.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from retailiq.core.enums import AppEnvironment, EmbeddingProvider, LLMProvider, LogFormat

# src/retailiq/core/settings.py -> core -> retailiq -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]

_BASE_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
    # Fields carry an explicit `validation_alias` so environment variable
    # names stay stable and readable. Without this flag that alias becomes
    # the *only* accepted key, and `Settings(retrieval={"top_k": 2})` would
    # be silently discarded — which is exactly how tests and dependency
    # injection construct these objects.
    populate_by_name=True,
)


class PathSettings(BaseSettings):
    """Filesystem locations for the knowledge base and persisted index."""

    model_config = _BASE_CONFIG

    project_root: Path = PROJECT_ROOT
    data_dir: Path = Field(default=PROJECT_ROOT / "data", validation_alias="DATA_DIR")
    chroma_dir: Path = Field(default=PROJECT_ROOT / "var" / "chroma", validation_alias="CHROMA_DIR")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def knowledge_base_dir(self) -> Path:
        """Directory of source markdown documents to ingest."""
        return self.data_dir / "knowledge_base"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def eval_dataset(self) -> Path:
        """Golden question set used by the evaluation harness."""
        return self.data_dir / "eval" / "eval_questions.json"


class LLMSettings(BaseSettings):
    """Chat-model provider selection and per-provider model names."""

    model_config = _BASE_CONFIG

    provider: LLMProvider = Field(default=LLMProvider.GROQ, validation_alias="LLM_PROVIDER")
    # Deterministic by default: graders must not wander between runs, or the
    # same question yields a different verdict each time and evaluation
    # numbers stop meaning anything.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, validation_alias="LLM_TEMPERATURE")
    request_timeout: int = Field(default=60, gt=0, validation_alias="LLM_REQUEST_TIMEOUT")
    max_retries: int = Field(default=2, ge=0, validation_alias="LLM_MAX_RETRIES")

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_CHAT_MODEL")

    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_chat_model: str = Field(
        default="llama-3.3-70b-versatile", validation_alias="GROQ_CHAT_MODEL"
    )

    google_api_key: SecretStr | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    google_chat_model: str = Field(default="gemini-1.5-flash", validation_alias="GOOGLE_CHAT_MODEL")

    ollama_chat_model: str = Field(default="llama3.1", validation_alias="OLLAMA_CHAT_MODEL")
    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL"
    )

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v

    def api_key_for(self, provider: LLMProvider) -> str | None:
        """Return the plain-text key for a provider, or None if unset."""
        secret = {
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.GOOGLE: self.google_api_key,
            LLMProvider.OLLAMA: None,  # local runtime, no key
        }.get(provider)
        return secret.get_secret_value() if secret else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def requires_api_key(self) -> bool:
        """Ollama runs locally; every hosted provider needs a key."""
        return self.provider is not LLMProvider.OLLAMA


class EmbeddingSettings(BaseSettings):
    """Embedding backend. Deliberately independent of the chat provider."""

    model_config = _BASE_CONFIG

    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.LOCAL, validation_alias="EMBED_PROVIDER"
    )
    openai_model: str = Field(
        default="text-embedding-3-small", validation_alias="OPENAI_EMBED_MODEL"
    )
    google_model: str = Field(
        default="models/text-embedding-004", validation_alias="GOOGLE_EMBED_MODEL"
    )
    ollama_model: str = Field(default="nomic-embed-text", validation_alias="OLLAMA_EMBED_MODEL")
    local_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="LOCAL_EMBED_MODEL"
    )

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v


class RetrievalSettings(BaseSettings):
    """Chunking and vector-search knobs."""

    model_config = _BASE_CONFIG

    collection_name: str = Field(default="retail_kb", validation_alias="COLLECTION_NAME")
    chunk_size: int = Field(default=800, gt=0, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, ge=0, validation_alias="CHUNK_OVERLAP")
    top_k: int = Field(default=4, gt=0, le=50, validation_alias="RETRIEVE_K")

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_fits(cls, v: int, info: Any) -> int:
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError(
                f"CHUNK_OVERLAP ({v}) must be smaller than CHUNK_SIZE ({size}); "
                "an overlap at or above the chunk size makes the splitter loop."
            )
        return v


class AgentSettings(BaseSettings):
    """Bounds on the self-correction loop.

    Every retry costs latency and tokens, so each corrective path gets an
    explicit budget. Without these the grader/rewrite cycle can ping-pong
    indefinitely on a question the knowledge base simply cannot answer.
    """

    model_config = _BASE_CONFIG

    max_query_rewrites: int = Field(default=2, ge=0, le=10, validation_alias="MAX_QUERY_REWRITES")
    max_generation_retries: int = Field(
        default=2, ge=0, le=10, validation_alias="MAX_GENERATION_RETRIES"
    )
    recursion_limit: int = Field(default=25, gt=0, validation_alias="RECURSION_LIMIT")


class ToolSettings(BaseSettings):
    """External tool availability."""

    model_config = _BASE_CONFIG

    enable_web_search: bool = Field(default=True, validation_alias="ENABLE_WEB_SEARCH")
    web_search_max_results: int = Field(
        default=4, gt=0, le=20, validation_alias="WEB_SEARCH_MAX_RESULTS"
    )


class ObservabilitySettings(BaseSettings):
    """LangSmith tracing.

    Off unless a key is supplied, so a fresh clone runs with no third-party
    account and sends no data anywhere. Tracing is what makes a multi-step
    agent debuggable in production: without the call tree, "why was this
    answer wrong?" is guesswork across six LLM calls.
    """

    model_config = _BASE_CONFIG

    tracing_enabled: bool = Field(default=False, validation_alias="LANGCHAIN_TRACING_V2")
    api_key: SecretStr | None = Field(default=None, validation_alias="LANGCHAIN_API_KEY")
    project: str = Field(default="retailiq", validation_alias="LANGCHAIN_PROJECT")
    endpoint: str = Field(
        default="https://api.smith.langchain.com", validation_alias="LANGCHAIN_ENDPOINT"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def api_key_value(self) -> str | None:
        """Plain-text key, or None. Kept off `__repr__` via `SecretStr`."""
        return self.api_key.get_secret_value() if self.api_key else None


class APISettings(BaseSettings):
    """HTTP server configuration."""

    model_config = _BASE_CONFIG

    host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    port: int = Field(default=8000, gt=0, lt=65536, validation_alias="API_PORT")
    # `NoDecode` suppresses pydantic-settings' automatic JSON parsing of
    # complex types from the environment. Without it, `API_CORS_ORIGINS=a,b`
    # fails as invalid JSON before the validator below ever runs.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"], validation_alias="API_CORS_ORIGINS"
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        """Accept ``a.com,b.com`` from the environment as a list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


class Settings(BaseSettings):
    """Root configuration object, composed of the sections above."""

    model_config = _BASE_CONFIG

    app_name: str = "RetailIQ"
    environment: AppEnvironment = Field(default=AppEnvironment.LOCAL, validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: LogFormat = Field(default=LogFormat.CONSOLE, validation_alias="LOG_FORMAT")

    paths: PathSettings = Field(default_factory=PathSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    api: APISettings = Field(default_factory=APISettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.environment is AppEnvironment.PRODUCTION

    def describe(self) -> dict[str, Any]:
        """Human-readable snapshot for diagnostics. Contains no secrets."""
        return {
            "app": self.app_name,
            "environment": str(self.environment),
            "llm_provider": str(self.llm.provider),
            "embed_provider": str(self.embeddings.provider),
            "collection": self.retrieval.collection_name,
            "top_k": self.retrieval.top_k,
            "chunk_size": self.retrieval.chunk_size,
            "chunk_overlap": self.retrieval.chunk_overlap,
            "max_query_rewrites": self.agent.max_query_rewrites,
            "web_search": self.tools.enable_web_search,
            "chroma_dir": str(self.paths.chroma_dir),
            "api_key_configured": self.llm.api_key_for(self.llm.provider) is not None,
            "tracing_enabled": self.observability.tracing_enabled
            and self.observability.api_key_value is not None,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the `.env` file is parsed once. Call `get_settings.cache_clear()`
    in tests that need to re-read a patched environment.
    """
    return Settings()
