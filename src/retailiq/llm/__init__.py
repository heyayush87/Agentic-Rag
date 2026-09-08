"""Model access layer.

The rest of the codebase never imports a provider SDK directly — it calls
`get_chat_model()` / `get_embeddings()` and the configured backend is
resolved here. Swapping OpenAI for Groq is a one-line `.env` change, and no
business logic mentions a vendor.
"""

from __future__ import annotations

from retailiq.llm.factory import clear_model_cache, get_chat_model, get_embeddings

__all__ = ["clear_model_cache", "get_chat_model", "get_embeddings"]
