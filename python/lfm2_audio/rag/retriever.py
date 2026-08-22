"""Retrieval pour l'outil ``search_knowledge_base`` (Phase 3)."""

from __future__ import annotations

import asyncio
from typing import Any


class KnowledgeBaseRetriever:
    def __init__(
        self,
        persist_dir: str,
        *,
        collection: str = "company_kb",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 4,
        max_chars_per_passage: int = 600,
    ) -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=persist_dir)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collection = client.get_collection(collection, embedding_function=ef)
        self.top_k = top_k
        self.max_chars_per_passage = max_chars_per_passage

    def search(self, query: str) -> list[dict[str, Any]]:
        res = self.collection.query(query_texts=[query], n_results=self.top_k)
        passages: list[dict[str, Any]] = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True):
            passages.append(
                {
                    "text": doc[: self.max_chars_per_passage],
                    "source": (meta or {}).get("source", "unknown"),
                    "score": round(1.0 - dist, 4),  # cosinus → similarité
                }
            )
        return passages

    async def asearch(self, query: str) -> list[dict[str, Any]]:
        """Interface async attendue par ``build_reception_registry``."""
        return await asyncio.to_thread(self.search, query)
