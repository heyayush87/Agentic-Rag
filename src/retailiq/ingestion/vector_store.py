"""Chroma vector store access.

Chroma was chosen because it persists to a local directory with no server to
run, which keeps `git clone && make ingest` working on any machine. The
store is reached only through this module, so migrating to pgvector,
Pinecone or Qdrant means rewriting this file and nothing else.
"""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from retailiq.core.exceptions import IngestionError, VectorStoreNotFoundError
from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings
from retailiq.llm.factory import get_embeddings

logger = get_logger(__name__)

# Opening a Chroma collection is not free, and the retriever is used on every
# request, so hold one handle per (directory, collection).
_store_cache: dict[tuple[str, str], Chroma] = {}


def _cache_key(settings: Settings) -> tuple[str, str]:
    return (str(settings.paths.chroma_dir), settings.retrieval.collection_name)


def reset_store_cache() -> None:
    """Drop cached handles — required after a rebuild swaps the files."""
    _store_cache.clear()


def index_exists(settings: Settings | None = None) -> bool:
    """True if a persisted index is present on disk."""
    settings = settings or get_settings()
    path = Path(settings.paths.chroma_dir)
    return path.exists() and any(path.iterdir())


def create_vector_store(chunks: list[Document], settings: Settings | None = None) -> Chroma:
    """Embed `chunks` and persist a fresh collection."""
    settings = settings or get_settings()
    if not chunks:
        raise IngestionError("Refusing to build an index from zero chunks.")

    directory = Path(settings.paths.chroma_dir)
    directory.mkdir(parents=True, exist_ok=True)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(settings),
        collection_name=settings.retrieval.collection_name,
        persist_directory=str(directory),
    )
    _store_cache[_cache_key(settings)] = store
    logger.info(
        "Vector store built",
        extra={"chunks": len(chunks), "path": str(directory)},
    )
    return store


def _remove_directory(directory: Path, attempts: int = 3) -> None:
    """Delete a directory, retrying while Windows releases file locks."""
    for attempt in range(1, attempts + 1):
        try:
            shutil.rmtree(directory)
            return
        except PermissionError:
            if attempt == attempts:
                break
            gc.collect()
            time.sleep(0.2 * attempt)

    # Never fail a rebuild over a stale lock: clear what we can, and let
    # `create_documents` overwrite the collection in place.
    shutil.rmtree(directory, ignore_errors=True)
    if directory.exists():
        logger.warning(
            "Could not fully remove the old index; it will be overwritten in place",
            extra={"path": str(directory)},
        )


def drop_index(settings: Settings | None = None) -> None:
    """Clear the persisted index. Used by `ingest --reset`.

    Drops the *collection* through Chroma's own API rather than deleting the
    directory. POSIX permits unlinking a file another handle still holds
    open; Windows does not, so once this process has served a single query
    the index files are memory-mapped and `rmtree` fails with
    ``PermissionError: [WinError 32]``. Deleting through the API sidesteps
    file locking entirely; removing the directory is kept only as a fallback
    for a corrupt or unreadable store.
    """
    settings = settings or get_settings()
    directory = Path(settings.paths.chroma_dir)

    if not directory.exists():
        reset_store_cache()
        return

    store = _store_cache.pop(_cache_key(settings), None)
    try:
        if store is None:
            store = Chroma(
                collection_name=settings.retrieval.collection_name,
                persist_directory=str(directory),
            )
        store.delete_collection()
        logger.info("Dropped existing collection", extra={"path": str(directory)})
    except Exception as exc:
        logger.debug(
            "Collection drop failed (%s); removing index files instead", type(exc).__name__
        )
        del store
        gc.collect()
        _remove_directory(directory)

    reset_store_cache()


def get_vector_store(settings: Settings | None = None) -> Chroma:
    """Open the persisted store for querying.

    Raises:
        VectorStoreNotFoundError: if ingestion has not been run. This is a
            503 rather than a 500 — the service is correctly configured but
            temporarily unable to serve until the index is built.
    """
    settings = settings or get_settings()
    key = _cache_key(settings)
    if key in _store_cache:
        return _store_cache[key]

    if not index_exists(settings):
        raise VectorStoreNotFoundError(str(settings.paths.chroma_dir))

    store = Chroma(
        collection_name=settings.retrieval.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.paths.chroma_dir),
    )
    _store_cache[key] = store
    return store


def get_retriever(settings: Settings | None = None, **overrides: Any) -> Any:
    """Return a top-k retriever over the persisted store."""
    settings = settings or get_settings()
    search_kwargs = {"k": settings.retrieval.top_k, **overrides}
    return get_vector_store(settings).as_retriever(search_kwargs=search_kwargs)
