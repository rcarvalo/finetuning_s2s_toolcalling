"""Ingestion de documents dans la base de connaissances ChromaDB.

Point d'entrée : ``lfm2-rag-ingest``.
La logique vit dans :mod:`lfm2_audio.rag.ingest` — ce module ne porte que la CLI.
"""

from __future__ import annotations

import argparse

from lfm2_audio.rag.ingest import (
    ingest,
)


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
