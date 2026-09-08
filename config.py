"""Central configuration. Reads from environment (.env) with sensible defaults."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = str(ROOT / "chroma_db")
COLLECTION_NAME = "retail_kb"

# --- Provider selection -------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").lower()

# --- Retrieval knobs ----------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "4"))

# --- Agent loop limits --------------------------------------------------
MAX_QUERY_REWRITES = int(os.getenv("MAX_QUERY_REWRITES", "2"))

# --- Web search fallback ------------------------------------------------
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
