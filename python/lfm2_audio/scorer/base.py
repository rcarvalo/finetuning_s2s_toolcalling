"""``BaseScorer`` — contrat commun à toutes les métriques.

Une seule abstraction sert la pipeline d'évaluation, la boucle d'entraînement et
l'analyse ad hoc : c'est ce qui évite de réimplémenter le WER une fois par
contexte d'appel.

:meth:`score` est une *template method* : elle traite la disponibilité, le hors
périmètre et les erreurs, de sorte qu'un scorer ne rate **jamais** une campagne.
Les sous-classes n'écrivent que :meth:`measure`, qui peut supposer que le
scorer est disponible et que l'échantillon le concerne.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample

logger = logging.getLogger(__name__)


class BaseScorer(ABC):
    """Métrique nommée, applicable à un :class:`EvalSample`."""

    name: str = ""
    """Identifiant stable — clé dans les rapports et les configs.

    Attribut de classe et non ``ClassVar`` : un substitut comme
    :class:`~lfm2_audio.scorer.missing.MissingScorer` doit pouvoir usurper le nom
    du scorer qu'il remplace, pour que la clé du rapport reste celle attendue.
    """

    higher_is_better: ClassVar[bool] = True
    """Sens de lecture. Faux pour le WER, vrai pour DNSMOS ou l'exactitude."""

    description: ClassVar[str] = ""

    # ------------------------------------------------------------------ #
    # Template method
    # ------------------------------------------------------------------ #

    def score(self, sample: EvalSample) -> ScoreResult:
        """Note un échantillon, sans jamais lever.

        Un scorer indisponible ou en erreur dégrade le rapport, il n'interrompt
        pas la campagne : une éval de 500 cas ne doit pas être perdue parce que
        le poids DNSMOS manque.
        """
        reason = self.unavailable_reason()
        if reason is not None:
            return ScoreResult.unavailable(self.name, reason)

        if not self.supports(sample):
            return ScoreResult.skipped(self.name, self.skip_reason(sample))

        try:
            return self.measure(sample)
        except Exception as error:  # un scorer en échec ne casse pas la campagne
            logger.warning("%s a échoué sur %s : %s", self.name, sample.sample_id, error)
            return ScoreResult.failed(self.name, f"{type(error).__name__}: {error}")

    def score_many(self, samples: Sequence[EvalSample]) -> list[ScoreResult]:
        """Note un lot. Surchargeable quand le batching change tout (ASR, GPU)."""
        return [self.score(sample) for sample in samples]

    # ------------------------------------------------------------------ #
    # À implémenter
    # ------------------------------------------------------------------ #

    @abstractmethod
    def measure(self, sample: EvalSample) -> ScoreResult:
        """Mesure effective. Peut supposer disponible et dans le périmètre."""

    # ------------------------------------------------------------------ #
    # Surchargeables
    # ------------------------------------------------------------------ #

    def unavailable_reason(self) -> str | None:
        """Raison de l'indisponibilité, ou ``None`` si le scorer peut tourner.

        Appelée à chaque échantillon : garder l'implémentation bon marché (les
        sous-classes mettent en cache leur sonde de dépendances).
        """
        return None

    def supports(self, sample: EvalSample) -> bool:
        """Vrai si l'échantillon entre dans le périmètre de ce scorer."""
        return True

    def skip_reason(self, sample: EvalSample) -> str:
        """Message associé à un ``SKIPPED``."""
        return "échantillon hors périmètre"

    # ------------------------------------------------------------------ #

    @property
    def is_available(self) -> bool:
        return self.unavailable_reason() is None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
