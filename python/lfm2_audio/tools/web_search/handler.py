"""Point d'entrée de l'outil ``web_search``, réinjecté en rôle ``tool``."""

from __future__ import annotations

from typing import Any

from lfm2_audio.tools.web_search.base import MAX_RESULTS, MAX_SNIPPET, WebSearchBackend, trim


async def web_search_handler(backend: WebSearchBackend, query: str) -> dict[str, Any]:
    """Cherche, puis **borne** le résultat avant de le rendre au modèle.

    Le bornage n'est pas cosmétique : réinjecter le pavé brut d'un backend web
    éloigne le tour ``tool`` de la distribution d'entraînement, et le modèle
    cesse d'ancrer sa réponse.
    """
    results = await backend.search(query)
    trimmed = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": trim(item.get("snippet", ""), MAX_SNIPPET),
        }
        for item in results[:MAX_RESULTS]
    ]
    return {"query": query, "results": trimmed}
