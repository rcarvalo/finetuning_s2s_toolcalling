"""Contrat d'un backend de recherche web, et bornage du résultat réinjecté.

Python pur — aucune dépendance réseau : le registre d'outils et l'orchestrateur
se testent donc sans rien installer.

Le tool_result d'ENTRAÎNEMENT (Phase B) était un petit dict compact. Réinjecter
le pavé brut d'un backend web (paragraphe Tavily + ``content`` de pages entières)
crée un décalage de distribution : le modèle ne sait plus ancrer sa réponse et
produit du vide ou du charabia. D'où les bornes ci-dessous.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

MAX_RESULTS = 2
MAX_SNIPPET = 160


@runtime_checkable
class WebSearchBackend(Protocol):
    """Rend une liste de ``{"title", "url", "snippet"}``."""

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]: ...


def trim(text: str, limit: int) -> str:
    """Tronque sur la dernière frontière de mot avant ``limit``."""
    stripped = (text or "").strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rsplit(" ", 1)[0] + "…"
