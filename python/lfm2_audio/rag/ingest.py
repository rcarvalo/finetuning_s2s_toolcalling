"""Ingestion de la base de connaissances entreprise dans ChromaDB (Phase 3).

Découpe les documents (``.md``/``.txt``) en chunks à recouvrement et les indexe
avec un modèle d'embeddings multilingue (FR).

Usage :
    python -m lfm2_audio.rag.ingest --docs ./kb_docs --persist-dir ./chroma_db
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 150) -> Iterator[str]:
    """Chunking par paragraphes avec taille cible et recouvrement (caractères)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
            continue
        if current:
            yield current
            current = current[-overlap:] if overlap else ""
        # paragraphe plus long que chunk_size : découpe dure
        remaining = para
        while len(remaining) > chunk_size:
            head = remaining[:chunk_size]
            yield (current + "\n\n" + head).strip() if current else head
            remaining = remaining[chunk_size - overlap :]
            current = ""
        current = f"{current}\n\n{remaining}".strip() if current else remaining
    if current.strip():
        yield current.strip()


def iter_documents(docs_dir: Path) -> Iterator[tuple[str, str]]:
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() in (".md", ".txt") and path.is_file():
            yield str(path.relative_to(docs_dir)), path.read_text(encoding="utf-8", errors="replace")


def ingest(
    docs_dir: str | Path,
    persist_dir: str | Path,
    *,
    collection: str = "company_kb",
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size: int = 800,
    overlap: int = 150,
) -> int:

    client = chromadb.PersistentClient(path=str(persist_dir))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    coll = client.get_or_create_collection(collection, embedding_function=ef, metadata={"hnsw:space": "cosine"})

    ids, documents, metadatas = [], [], []
    for source, text in iter_documents(Path(docs_dir)):
        for i, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            ids.append(f"{source}::chunk{i}")
            documents.append(chunk)
            metadatas.append({"source": source, "chunk": i})

    if not ids:
        return 0

    # Upsert par lots (limite de batch ChromaDB)
    for start in range(0, len(ids), 256):
        sl = slice(start, start + 256)
        coll.upsert(ids=ids[sl], documents=documents[sl], metadatas=metadatas[sl])
    return len(ids)
