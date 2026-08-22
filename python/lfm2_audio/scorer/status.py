"""Issue d'une tentative de scoring."""

from __future__ import annotations

from enum import StrEnum


class ScoreStatus(StrEnum):
    """Ce qui est arrivé à un scorer sur un échantillon.

    Distinguer ``UNAVAILABLE`` de ``FAILED`` compte : le premier est une absence
    de dépendance (poids DNSMOS manquants, pas de clé d'API) et ne dit rien du
    modèle évalué ; le second est un échec sur cet échantillon précis. Les
    confondre ferait passer une éval partielle pour une éval complète.
    """

    OK = "ok"
    """Mesure effectuée."""

    UNAVAILABLE = "unavailable"
    """Dépendance absente — le scorer n'a pas pu s'exécuter du tout."""

    SKIPPED = "skipped"
    """Échantillon hors périmètre (pas d'audio à noter, pas de référence…)."""

    FAILED = "failed"
    """Erreur pendant la mesure de cet échantillon."""

    @property
    def is_measurement(self) -> bool:
        """Vrai si le statut porte une valeur agrégeable."""
        return self is ScoreStatus.OK
