"""End-to-end ingestion against a real Chroma store with fake embeddings.

Uses a deterministic hash-based embedding so the pipeline — load, chunk,
embed, persist, retrieve — is exercised for real without downloading a model
or calling an API.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retailiq.core.exceptions import VectorStoreNotFoundError
from retailiq.core.settings import Settings

pytestmark = pytest.mark.integration


class DeterministicEmbeddings:
    """Tiny character-frequency embedding. Not good, but stable and free."""

    DIMENSIONS = 64

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.DIMENSIONS
        for char in text.lower():
            vector[ord(char) % self.DIMENSIONS] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def settings(tmp_path: Path, knowledge_base: Path) -> Settings:
    return Settings(
        paths={"data_dir": tmp_path / "data", "chroma_dir": tmp_path / "chroma"},  # type: ignore[arg-type]
        retrieval={"chunk_size": 200, "chunk_overlap": 20, "top_k": 2},  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    from retailiq.ingestion import vector_store

    monkeypatch.setattr(vector_store, "get_embeddings", lambda *a, **k: DeterministicEmbeddings())
    vector_store.reset_store_cache()


def test_full_pipeline_builds_a_searchable_index(settings: Settings) -> None:
    from retailiq.ingestion.pipeline import build_index
    from retailiq.ingestion.vector_store import get_retriever

    result = build_index(reset=True, settings=settings)

    assert result.documents == 2
    assert result.chunks >= 2
    assert Path(settings.paths.chroma_dir).exists()

    retrieved = get_retriever(settings).invoke("electrical returns")
    assert retrieved
    assert all("source" in d.metadata for d in retrieved)


def test_querying_before_ingestion_raises_a_typed_error(settings: Settings) -> None:
    from retailiq.ingestion.vector_store import get_retriever

    with pytest.raises(VectorStoreNotFoundError):
        get_retriever(settings)


def test_reset_replaces_rather_than_appends(settings: Settings) -> None:
    """Chroma appends by default; a re-ingest without reset duplicates chunks
    and fills top-k with copies of the same passage."""
    from retailiq.ingestion.pipeline import build_index
    from retailiq.ingestion.vector_store import get_vector_store

    build_index(reset=True, settings=settings)
    first = get_vector_store(settings)._collection.count()

    build_index(reset=True, settings=settings)
    second = get_vector_store(settings)._collection.count()

    assert first == second
