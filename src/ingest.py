"""Ingestion: load the retail docs, chunk them, and embed into a local Chroma store.

Run once (or after changing the docs):
    python -m src.ingest
"""
from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import config
from src.llm import get_embeddings


def load_documents():
    """Load every markdown file in the data directory, tagging each with a source."""
    docs = []
    for md_path in sorted(Path(config.DATA_DIR).glob("*.md")):
        loaded = TextLoader(str(md_path), encoding="utf-8").load()
        for d in loaded:
            d.metadata["source"] = md_path.name
            d.metadata["topic"] = md_path.stem
        docs.extend(loaded)
    if not docs:
        raise FileNotFoundError(f"No .md files found in {config.DATA_DIR}")
    return docs


def build_vectorstore(reset: bool = True) -> Chroma:
    """Chunk documents and (re)build the Chroma vector store on disk."""
    if reset and Path(config.CHROMA_DIR).exists():
        shutil.rmtree(config.CHROMA_DIR)

    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
    )
    print(f"Ingested {len(docs)} documents -> {len(chunks)} chunks into {config.CHROMA_DIR}")
    return vectordb


def get_vectorstore() -> Chroma:
    """Open the existing persisted store (used at query time)."""
    if not Path(config.CHROMA_DIR).exists():
        raise FileNotFoundError(
            "Vector store not found. Run `python -m src.ingest` first."
        )
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": config.RETRIEVE_K})


if __name__ == "__main__":
    build_vectorstore(reset=True)
