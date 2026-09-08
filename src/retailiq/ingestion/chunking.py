"""Split documents into retrievable chunks.

Why these settings
------------------
**Separator order (`\\n## `, `\\n### `, blank line, newline, space).** The
knowledge base is structured markdown where each heading is a self-contained
policy. Splitting on headings first means a chunk tends to be one whole rule
("Electrical returns: 30 days") rather than half of two adjacent rules. A
chunk that splits mid-policy is the single most common cause of a confidently
wrong RAG answer.

**Chunk size 800.** Large enough to hold a complete policy with its
qualifiers, small enough that four retrieved chunks stay well inside the
context window and the relevance grader has a cheap unit to judge.

**Overlap 120 (~15%).** Insurance against a fact landing exactly on a
boundary. Too little and boundary facts get lost; too much and you pay to
embed and grade duplicated text.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retailiq.core.logging import get_logger
from retailiq.core.settings import Settings, get_settings

logger = get_logger(__name__)

#: Tried in order; the splitter falls back down the list until chunks fit.
MARKDOWN_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]


def build_splitter(settings: Settings | None = None) -> RecursiveCharacterTextSplitter:
    """Construct the configured text splitter."""
    settings = settings or get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.retrieval.chunk_size,
        chunk_overlap=settings.retrieval.chunk_overlap,
        separators=MARKDOWN_SEPARATORS,
        add_start_index=True,  # lets a chunk be located back in its source file
    )


def chunk_documents(documents: list[Document], settings: Settings | None = None) -> list[Document]:
    """Split documents into chunks, preserving each document's metadata."""
    settings = settings or get_settings()
    chunks = build_splitter(settings).split_documents(documents)

    logger.info(
        "Chunked documents",
        extra={
            "documents": len(documents),
            "chunks": len(chunks),
            "chunk_size": settings.retrieval.chunk_size,
            "chunk_overlap": settings.retrieval.chunk_overlap,
        },
    )
    return chunks
