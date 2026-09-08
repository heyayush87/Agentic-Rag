"""Load source documents and attach provenance metadata.

Provenance is not decoration. The generator is instructed to cite its
sources, and the evaluation harness scores whether the *expected* document
reached the context — both depend on every chunk carrying a `source` tag
from the moment it is loaded.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from retailiq.core.exceptions import IngestionError
from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings

logger = get_logger(__name__)

#: Extensions understood by the loader today. PDFs and HTML slot in here
#: by adding a branch — the rest of the pipeline is format-agnostic.
SUPPORTED_SUFFIXES = frozenset({".md", ".markdown", ".txt"})


def discover_documents(directory: Path) -> list[Path]:
    """Return supported files in `directory`, sorted for deterministic order.

    Determinism matters: identical inputs must produce an identical index,
    or two runs of the evaluation harness are not comparable.
    """
    if not directory.exists():
        raise IngestionError(
            f"Knowledge base directory not found: {directory}", directory=str(directory)
        )
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)


def _display_path(path: Path, root: Path) -> str:
    """Path relative to the project root, or absolute if it lies outside it.

    `Path.relative_to` raises when the target is not under `root`, which is
    the normal case for a container bind-mount or an absolute `DATA_DIR`.
    Metadata for a log line must never be able to fail ingestion.
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_knowledge_base(settings: Settings | None = None) -> list[Document]:
    """Load every knowledge-base document with source metadata attached."""
    settings = settings or get_settings()
    directory = settings.paths.knowledge_base_dir

    paths = discover_documents(directory)
    if not paths:
        raise IngestionError(
            f"No ingestible documents in {directory}. "
            f"Expected files with extensions: {', '.join(sorted(SUPPORTED_SUFFIXES))}.",
            directory=str(directory),
        )

    documents: list[Document] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise IngestionError(f"Could not read {path.name}: {exc}", path=str(path)) from exc

        if not text.strip():
            logger.warning("Skipping empty document", extra={"source": path.name})
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "topic": path.stem,
                    "path": _display_path(path, settings.paths.project_root),
                },
            )
        )

    logger.info("Loaded knowledge base", extra={"documents": len(documents)})
    return documents
