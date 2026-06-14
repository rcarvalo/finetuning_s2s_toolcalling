"""Câblage du registre des outils anglais ``web_search`` + ``db_query`` (v1).

Assemble le ``ToolRegistry`` dont les ``definitions()`` servent à la fois
l'orchestrateur (inférence) ET la génération des données d'entraînement
(contrat unique entraînement/inférence, comme ``build_reception_registry``).

``db_query`` prend une QUESTION en langage naturel ; la traduction NL→SQL est
faite côté backend (déterministe/templaté sur ``sql/schema_en.sql``), hors du
chemin du modèle assistant. Le ``StubDbQueryBackend`` suffit pour les tests et
la synthèse de données (la v1 single-turn n'exécute aucun outil).
"""

from __future__ import annotations

from typing import Any, Protocol

from s2s_toolcalling.tools import schemas
from s2s_toolcalling.tools.registry import ToolRegistry
from s2s_toolcalling.tools.web_search import (
    StubWebSearchBackend,
    WebSearchBackend,
    web_search_handler,
)


class DbQueryBackend(Protocol):
    async def answer(self, question: str) -> dict[str, Any]: ...


class StubDbQueryBackend:
    """Réponse canned déterministe à une question NL (tests / synthèse)."""

    async def answer(self, question: str) -> dict[str, Any]:
        return {
            "question": question,
            "answer": f"(stub) answer to: {question}",
            "rows": [],
        }


def build_toolcalling_en_registry(
    *,
    web_backend: WebSearchBackend | None = None,
    db_backend: DbQueryBackend | None = None,
    timeout_s: float = 10.0,
) -> ToolRegistry:
    """Registre des 2 outils EN. Backends stub par défaut (tests / synthèse)."""
    web = web_backend or StubWebSearchBackend()
    db = db_backend or StubDbQueryBackend()
    registry = ToolRegistry(timeout_s=timeout_s)

    async def web_search(query: str) -> dict[str, Any]:
        return await web_search_handler(web, query)

    async def db_query(question: str) -> dict[str, Any]:
        return await db.answer(question)

    registry.register(schemas.WEB_SEARCH, web_search)
    registry.register(schemas.DB_QUERY, db_query)
    return registry
