"""Provider-agnostic factory for chat models and embeddings.

Design notes
------------
**Why a factory at all?** The agent makes six or seven LLM calls per question
(route, grade each chunk, generate, check grounding, check usefulness). If
each site constructed its own client, changing provider would mean touching
every one. Here it is one dispatch table.

**Why is chat separate from embeddings?** They are genuinely independent
choices. Groq serves chat faster than anyone but exposes *no* embedding
endpoint, so the natural pairing is Groq chat + on-device sentence-transformers.
Keeping the two settings apart makes that combination expressible.

**Why cache the instances?** Constructing an embedding model downloads and
loads weights (~90 MB for MiniLM). Doing that per request would dominate
latency. Clients are stateless and safe to share.
"""

from __future__ import annotations

from typing import Any

from retailiq.core.enums import EmbeddingProvider, LLMProvider
from retailiq.core.exceptions import ConfigurationError, ProviderNotInstalledError
from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings

logger = get_logger(__name__)

# Keyed by the settings that affect construction, so a test that switches
# provider is not served a stale client.
_chat_cache: dict[tuple[Any, ...], Any] = {}
_embed_cache: dict[tuple[Any, ...], Any] = {}


def clear_model_cache() -> None:
    """Drop cached clients. Used by tests and after a config reload."""
    _chat_cache.clear()
    _embed_cache.clear()


def _require_api_key(settings: Settings, provider: LLMProvider) -> None:
    """Fail fast with an actionable message rather than a 401 mid-graph."""
    if not settings.llm.requires_api_key:
        return
    if settings.llm.api_key_for(provider):
        return
    env_var = f"{provider.value.upper()}_API_KEY"
    raise ConfigurationError(
        f"LLM_PROVIDER={provider.value!r} but {env_var} is not set. "
        f"Add it to your .env file, or switch to LLM_PROVIDER=ollama to run fully locally.",
        provider=provider.value,
        missing_env_var=env_var,
    )


def get_chat_model(temperature: float | None = None, settings: Settings | None = None) -> Any:
    """Return a LangChain chat model for the configured provider.

    Args:
        temperature: Override the configured sampling temperature. Graders
            leave this at 0 for reproducibility; only creative generation
            has any reason to raise it.
        settings: Injected configuration. Defaults to the global singleton.
    """
    settings = settings or get_settings()
    provider = settings.llm.provider
    temp = settings.llm.temperature if temperature is None else temperature

    cache_key = ("chat", provider, temp, settings.llm.request_timeout)
    if cache_key in _chat_cache:
        return _chat_cache[cache_key]

    _require_api_key(settings, provider)
    logger.debug("Constructing chat model", extra={"provider": str(provider), "temperature": temp})

    model = _build_chat_model(settings, provider, temp)
    _chat_cache[cache_key] = model
    return model


def _build_chat_model(settings: Settings, provider: LLMProvider, temperature: float) -> Any:
    """Dispatch to the selected provider, translating missing SDKs."""
    common = {"temperature": temperature, "timeout": settings.llm.request_timeout}

    if provider is LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ProviderNotInstalledError("openai", "langchain-openai", "openai") from exc
        return ChatOpenAI(
            model=settings.llm.openai_chat_model,
            api_key=settings.llm.openai_api_key,
            max_retries=settings.llm.max_retries,
            **common,
        )

    if provider is LLMProvider.GROQ:
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ProviderNotInstalledError("groq", "langchain-groq", "groq") from exc
        return ChatGroq(
            model=settings.llm.groq_chat_model,
            api_key=settings.llm.groq_api_key,
            max_retries=settings.llm.max_retries,
            **common,
        )

    if provider is LLMProvider.GOOGLE:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ProviderNotInstalledError("google", "langchain-google-genai", "google") from exc
        key = settings.llm.google_api_key
        return ChatGoogleGenerativeAI(
            model=settings.llm.google_chat_model,
            google_api_key=key.get_secret_value() if key else None,
            temperature=temperature,
            max_retries=settings.llm.max_retries,
        )

    if provider is LLMProvider.OLLAMA:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ProviderNotInstalledError("ollama", "langchain-ollama", "ollama") from exc
        return ChatOllama(
            model=settings.llm.ollama_chat_model,
            base_url=settings.llm.ollama_base_url,
            temperature=temperature,
        )

    raise ConfigurationError(
        f"Unsupported LLM_PROVIDER={provider!r}. "
        f"Choose one of: {', '.join(p.value for p in LLMProvider)}."
    )


def get_embeddings(settings: Settings | None = None) -> Any:
    """Return the embedding model for the configured `EMBED_PROVIDER`.

    Critical invariant: the model used to build the index and the model used
    at query time must be identical. Different models produce vectors in
    incompatible spaces, so changing `EMBED_PROVIDER` requires a full
    re-ingest — searching a MiniLM index with OpenAI vectors returns noise,
    silently and without error.
    """
    settings = settings or get_settings()
    provider = settings.embeddings.provider

    cache_key = ("embed", provider, _embedding_model_name(settings))
    if cache_key in _embed_cache:
        return _embed_cache[cache_key]

    logger.debug("Constructing embeddings", extra={"provider": str(provider)})
    model = _build_embeddings(settings, provider)
    _embed_cache[cache_key] = model
    return model


def _embedding_model_name(settings: Settings) -> str:
    """The configured model name for the active embedding provider."""
    return {
        EmbeddingProvider.OPENAI: settings.embeddings.openai_model,
        EmbeddingProvider.GOOGLE: settings.embeddings.google_model,
        EmbeddingProvider.OLLAMA: settings.embeddings.ollama_model,
        EmbeddingProvider.LOCAL: settings.embeddings.local_model,
    }[settings.embeddings.provider]


def _build_embeddings(settings: Settings, provider: EmbeddingProvider) -> Any:
    if provider is EmbeddingProvider.OPENAI:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise ProviderNotInstalledError("openai", "langchain-openai", "openai") from exc
        return OpenAIEmbeddings(
            model=settings.embeddings.openai_model,
            api_key=settings.llm.openai_api_key,
        )

    if provider is EmbeddingProvider.GOOGLE:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ProviderNotInstalledError("google", "langchain-google-genai", "google") from exc
        key = settings.llm.google_api_key
        return GoogleGenerativeAIEmbeddings(
            model=settings.embeddings.google_model,
            google_api_key=key.get_secret_value() if key else None,
        )

    if provider is EmbeddingProvider.OLLAMA:
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise ProviderNotInstalledError("ollama", "langchain-ollama", "ollama") from exc
        return OllamaEmbeddings(
            model=settings.embeddings.ollama_model,
            base_url=settings.llm.ollama_base_url,
        )

    if provider is EmbeddingProvider.LOCAL:
        # Runs on-device, no API key, no per-token cost. `langchain_huggingface`
        # is the maintained home for this class; fall back to the deprecated
        # community location so older installs keep working.
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError as exc:
                raise ProviderNotInstalledError("local", "langchain-huggingface", "local") from exc
        return HuggingFaceEmbeddings(model_name=settings.embeddings.local_model)

    raise ConfigurationError(
        f"Unsupported EMBED_PROVIDER={provider!r}. "
        f"Choose one of: {', '.join(p.value for p in EmbeddingProvider)}."
    )
