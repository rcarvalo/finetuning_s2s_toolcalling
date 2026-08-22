"""``TavilyBackend`` — recherche web optimisée LLM (clé ``TAVILY_API_KEY``).

Contrairement à DuckDuckGo text, qui renvoie des pages génériques médiocres pour
répondre à une question factuelle, Tavily rend du contenu propre et une réponse
synthétique directement exploitable par le modèle.

``tavily`` est importé en tête : ce module n'est chargé que si on construit ce
backend.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from tavily import TavilyClient


class TavilyBackend:
    """Recherche via l'API Tavily."""

    def __init__(self, *, max_results: int = 4, api_key: str | None = None, depth: str = "basic") -> None:
        self.max_results = max_results
        self.api_key = api_key
        self.depth = depth

    async def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._search_sync, query, max_results or self.max_results)

    def _search_sync(self, query: str, count: int) -> list[dict[str, Any]]:
        client = TavilyClient(api_key=self.api_key or os.environ["TAVILY_API_KEY"])
        response = client.search(query, max_results=count, search_depth=self.depth, include_answer=True)

        results: list[dict[str, Any]] = []
        if response.get("answer"):
            # Réponse synthétique en tête : c'est elle que le modèle prononcera.
            results.append({"title": "answer", "url": "", "snippet": response["answer"]})
        results += [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in response.get("results", [])
        ]
        return results
