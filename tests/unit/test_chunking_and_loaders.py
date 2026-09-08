"""Ingestion units: document loading and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from retailiq.core.exceptions import IngestionError
from retailiq.core.settings import Settings
from retailiq.ingestion.chunking import chunk_documents
from retailiq.ingestion.loaders import discover_documents, load_knowledge_base

pytestmark = pytest.mark.unit


def _settings(tmp_path: Path, **retrieval: object) -> Settings:
    kwargs: dict[str, object] = {
        "paths": {"data_dir": tmp_path / "data", "chroma_dir": tmp_path / "chroma"}
    }
    if retrieval:
        kwargs["retrieval"] = retrieval
    return Settings(**kwargs)  # type: ignore[arg-type]


def test_documents_load_with_source_metadata(tmp_path: Path, knowledge_base: Path) -> None:
    """Provenance drives citation and evaluation; it must exist from load time."""
    documents = load_knowledge_base(_settings(tmp_path))

    assert len(documents) == 2
    sources = {d.metadata["source"] for d in documents}
    assert sources == {"returns_policy.md", "loyalty_clubcard.md"}
    assert all(d.metadata["topic"] for d in documents)


def test_discovery_is_deterministic(tmp_path: Path, knowledge_base: Path) -> None:
    """Identical inputs must yield an identical index across runs."""
    assert discover_documents(knowledge_base) == discover_documents(knowledge_base)


def test_missing_directory_raises_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="not found"):
        discover_documents(tmp_path / "nope")


def test_empty_knowledge_base_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "data" / "knowledge_base").mkdir(parents=True)
    with pytest.raises(IngestionError, match="No ingestible documents"):
        load_knowledge_base(_settings(tmp_path))


def test_chunking_preserves_source_metadata(tmp_path: Path) -> None:
    """A chunk that loses its source cannot be cited."""
    documents = [
        Document(
            page_content="## A\n" + ("word " * 400) + "\n## B\n" + ("other " * 400),
            metadata={"source": "returns_policy.md", "topic": "returns"},
        )
    ]
    chunks = chunk_documents(documents, _settings(tmp_path, chunk_size=200, chunk_overlap=20))

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "returns_policy.md" for c in chunks)


def test_chunks_respect_the_configured_size(tmp_path: Path) -> None:
    documents = [Document(page_content="word " * 2000, metadata={"source": "x.md"})]
    chunks = chunk_documents(documents, _settings(tmp_path, chunk_size=300, chunk_overlap=30))

    # RecursiveCharacterTextSplitter can overshoot slightly on unsplittable
    # runs, so allow a small tolerance rather than asserting a hard ceiling.
    assert all(len(c.page_content) <= 400 for c in chunks)
