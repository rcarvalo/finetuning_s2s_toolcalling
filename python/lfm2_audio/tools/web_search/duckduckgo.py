"""``DuckDuckGoBackend`` — recherche web sans clé d'API.

Repli quand ``TAVILY_API_KEY`` est absent : les pages rendues sont génériques et
médiocres pour répondre à une question factuelle. Préférer Tavily en démo.

``ddgs`` est importé en tête : ce module n'est chargé que si on construit ce
backend.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ddgs import DDGS


class DuckDuckGoBackend:
    """Recherche via ``ddgs``, exécutée hors de la boucle asyncio."""

    def __init__(self, *, max_results: int = 5, region: str = "wt-wt", safesearch: str = "moderate") -> None:
        self.max_results = max_results
        self.region = region
        self.safesearch = safesearch

    async def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        # ddgs est synchrone : le déporter évite de bloquer la boucle.
        return await asyncio.to_thread(self._search_sync, query, max_results or self.max_results)

    def _search_sync(self, query: str, count: int) -> list[dict[str, Any]]:
        with DDGS() as client:
            hits = client.text(query, region=self.region, safesearch=self.safesearch, max_results=count)
        return [{"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")} for h in hits]
