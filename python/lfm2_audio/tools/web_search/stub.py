"""``StubWebSearchBackend`` — résultats déterministes, sans réseau.

Sert aux tests et à la génération de données synthétiques : les réponses d'outil
réinjectées en Phase B n'ont pas besoin d'être réelles, seulement bien formées.
"""

from __future__ import annotations

from typing import Any


class StubWebSearchBackend:
    """Résultats fabriqués à partir de la requête."""

    def __init__(self, *, max_results: int = 3) -> None:
        self.max_results = max_results

    async def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        count = max_results or self.max_results
        cleaned = query.strip()
        return [
            {
                "title": f"Result {index + 1} for “{cleaned}”",
                "url": f"https://example.com/search?q={cleaned.replace(' ', '+')}&r={index + 1}",
                "snippet": f"A relevant passage about {cleaned} (stub result {index + 1}).",
            }
            for index in range(count)
        ]
