"""``Judge`` — contrat minimal d'un juge LLM.

Un ``Protocol`` : le scorer de raisonnement dépend du fait d'être jugé, pas de
Gemini. Un juge factice satisfait ce contrat, ce qui rend le scorer testable
sans appel réseau ni clé d'API.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    """Répond à un prompt de notation par du texte (JSON attendu)."""

    def judge(self, prompt: str) -> str: ...
