"""``MissingScorer`` — substitut d'un scorer dont les dépendances manquent.

*Null object* : il occupe la place du scorer absent et rend systématiquement
``UNAVAILABLE`` avec la raison. La pipeline n'a donc aucun cas particulier à
traiter, et le rapport distingue « non mesuré faute d'outillage » de « mesuré et
mauvais » — deux conclusions très différentes.
"""

from __future__ import annotations

from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample


class MissingScorer(BaseScorer):
    """Scorer inopérant, qui explique pourquoi.

    Le nom est posé sur l'instance : il usurpe celui du scorer absent, pour que
    le rapport garde la clé attendue (`wer`, `dnsmos`…) plutôt qu'un générique.
    """

    def __init__(self, name: str, reason: str) -> None:
        self.name = name  # usurpe le nom du scorer absent : la clé du rapport ne bouge pas
        self._reason = reason

    def unavailable_reason(self) -> str | None:
        return self._reason

    def measure(self, sample: EvalSample) -> ScoreResult:
        # Jamais atteint : score() court-circuite sur unavailable_reason().
        return ScoreResult.unavailable(self.name, self._reason)
