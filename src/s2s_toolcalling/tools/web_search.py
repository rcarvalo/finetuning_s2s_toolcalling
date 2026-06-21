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


class TavilyBackend:
    """Recherche web optimisée LLM via Tavily (clé ``TAVILY_API_KEY``).

    Contrairement à DuckDuckGo text (qui renvoie des pages génériques, médiocres
    pour répondre à une question factuelle), Tavily renvoie du contenu propre et
    pertinent + une réponse synthétique (``answer``), directement exploitable par
    le modèle. Recommandé pour la démo « questionner le web ».
    """

    def __init__(self, *, max_results: int = 4, api_key: str | None = None, depth: str = "basic") -> None:
        self.max_results = max_results
        self.api_key = api_key
        self.depth = depth

    async def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        import asyncio

        return await asyncio.to_thread(self._search_sync, query, max_results or self.max_results)

    def _search_sync(self, query: str, k: int) -> list[dict[str, Any]]:
        import os

        from tavily import TavilyClient

        resp = TavilyClient(api_key=self.api_key or os.environ["TAVILY_API_KEY"]).search(
            query, max_results=k, search_depth=self.depth, include_answer=True
        )
        out: list[dict[str, Any]] = []
        if resp.get("answer"):  # réponse synthétique en 1ère position (le modèle la parle)
            out.append({"title": "answer", "url": "", "snippet": resp["answer"]})
        out += [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                for r in resp.get("results", [])]
        return out


class DuckDuckGoBackend:
    """Recherche web réelle via ``ddgs`` (paresseux) — pages génériques (médiocre
    pour répondre). Préférer ``TavilyBackend`` pour la démo."""

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


# Le tool_result d'ENTRAÎNEMENT (Phase B) était un petit dict compact. Réinjecter
# le pavé brut d'un backend web (paragraphe Tavily + `content` de pages entières)
# crée un décalage de distribution → le modèle ne sait plus grounder (réponse vide
# ou charabia). On BORNE donc le résultat réinjecté : peu de hits, snippets courts.
_MAX_RESULTS = 3
_MAX_SNIPPET = 280


def _trim(text: str, limit: int) -> str:
    s = (text or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit].rsplit(" ", 1)[0] + "…"


async def web_search_handler(backend: WebSearchBackend, query: str) -> dict[str, Any]:
    """Point d'entrée de l'outil ``web_search`` (réinjecté en rôle ``tool``).

    Résultat BORNÉ (``_MAX_RESULTS`` hits, snippets ``_MAX_SNIPPET`` car.) pour
    rester proche de la distribution d'entraînement et fiabiliser le grounding.
    """
    results = await backend.search(query)
    trimmed = [
        {"title": r.get("title", ""), "url": r.get("url", ""),
         "snippet": _trim(r.get("snippet", ""), _MAX_SNIPPET)}
        for r in results[:_MAX_RESULTS]
    ]
    return {"query": query, "results": trimmed}
