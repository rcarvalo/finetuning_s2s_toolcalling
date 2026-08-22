"""Recherche web : contrat, handler, et une implémentation par fichier.

Chaque backend importe sa dépendance **en tête** (``tavily``, ``ddgs``) et n'est
donc chargé que si on le construit. ``base``, ``stub`` et ``handler`` restent en
Python pur : l'orchestrateur et son registre se testent sans réseau.
"""

from lfm2_audio.tools.web_search.base import (
    MAX_RESULTS,
    MAX_SNIPPET,
    WebSearchBackend,
    trim,
)
from lfm2_audio.tools.web_search.handler import web_search_handler
from lfm2_audio.tools.web_search.stub import StubWebSearchBackend

__all__ = [
    "MAX_RESULTS",
    "MAX_SNIPPET",
    "StubWebSearchBackend",
    "WebSearchBackend",
    "trim",
    "web_search_handler",
]
