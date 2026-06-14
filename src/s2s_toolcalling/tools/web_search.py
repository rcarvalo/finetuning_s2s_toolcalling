"""Backends de l'outil ``web_search`` (capacité tool calling vocal EN).

Deux implémentations, même contrat ``WebSearchBackend.search`` :
- ``StubWebSearchBackend`` : résultats déterministes dérivés de la requête —
  aucun réseau. Utilisé pour les tests et la génération de données synthétiques
  (les réponses d'outil réinjectées en Phase B n'ont pas besoin d'être réelles).
- ``DuckDuckGoBackend`` : recherche web réelle via ``ddgs`` (import paresseux,
  extra ``serve``) — branché par l'orchestrateur en production.

Le handler renvoie ``{"results": [{"title", "url", "snippet"}, ...]}`` : c'est ce
payload qui est réinjecté au modèle via ``chat_format.render_tool_response``.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class WebSearchBackend(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]: ...


class StubWebSearchBackend:
    """Résultats canned déterministes (tests / synthèse de données, sans réseau)."""

    def __init__(self, *, max_results: int = 3) -> None:
        self.max_results = max_results

    async def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        k = max_results or self.max_results
        q = query.strip()
        return [
            {
                "title": f"Result {i + 1} for “{q}”",
                "url": f"https://example.com/search?q={q.replace(' ', '+')}&r={i + 1}",
                "snippet": f"A relevant passage about {q} (stub result {i + 1}).",
            }
            for i in range(k)
        ]


class DuckDuckGoBackend:
    """Recherche web réelle via ``ddgs`` (paresseux). Backend de production."""

    def __init__(self, *, max_results: int = 5, region: str = "wt-wt", safesearch: str = "moderate") -> None:
        self.max_results = max_results
        self.region = region
        self.safesearch = safesearch

    async def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        import asyncio

        k = max_results or self.max_results
        # ddgs est synchrone → exécuté hors de la boucle pour ne pas bloquer.
        return await asyncio.to_thread(self._search_sync, query, k)

    def _search_sync(self, query: str, k: int) -> list[dict[str, Any]]:
        from ddgs import DDGS

        with DDGS() as ddgs:
            hits = ddgs.text(query, region=self.region, safesearch=self.safesearch, max_results=k)
        return [
            {"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")}
            for h in hits
        ]


async def web_search_handler(backend: WebSearchBackend, query: str) -> dict[str, Any]:
    """Point d'entrée de l'outil ``web_search`` (réinjecté en rôle ``tool``)."""
    results = await backend.search(query)
    return {"query": query, "results": results}
