"""Provider-flexible factory for chat models and embeddings.

The whole point of this module: the rest of the codebase never imports a
specific provider. It calls `get_chat_model()` / `get_embeddings()` and the
right backend is chosen from environment config. That makes the project
runnable today whether the user has an OpenAI key, a free Groq/Gemini key,
or nothing at all (local Ollama + local embeddings).
"""
from __future__ import annotations

import os

import config


def get_chat_model(temperature: float = 0.0):
    """Return a LangChain chat model for the configured provider."""
    provider = config.LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5-flash"),
            temperature=temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use one of: openai, groq, google, ollama."
    )


def get_embeddings():
    """Return an embeddings model for the configured EMBED_PROVIDER.

    Groq has no embedding endpoint, so its natural companion is `local`
    (sentence-transformers), which keeps the whole pipeline free.
    """
    provider = config.EMBED_PROVIDER

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        )

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if provider == "local":
        # Runs on-device with no API key. Small, fast, good enough for a demo.
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )

    raise ValueError(
        f"Unknown EMBED_PROVIDER={provider!r}. Use one of: openai, google, ollama, local."
    )
