"""Ingestion de la base de connaissances entreprise dans ChromaDB (Phase 3).

Découpe les documents (``.md``/``.txt``) en chunks à recouvrement et les indexe
avec un modèle d'embeddings multilingue (FR).

Usage :
    python -m s2s_toolcalling.rag.ingest --docs ./kb_docs --persist-dir ./chroma_db
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator


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
        while len(para) > chunk_size:
            yield (current + "\n\n" + para[:chunk_size]).strip() if current else para[:chunk_size]
            para = para[chunk_size - overlap :]
            current = ""
        current = f"{current}\n\n{para}".strip() if current else para
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
    import chromadb
    from chromadb.utils import embedding_functions

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", required=True, help="Répertoire des documents (.md/.txt)")
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--collection", default="company_kb")
    parser.add_argument("--embedding-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    n = ingest(
        args.docs,
        args.persist_dir,
        collection=args.collection,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"{n} chunks indexés dans {args.persist_dir} (collection {args.collection})")


if __name__ == "__main__":
    main()
