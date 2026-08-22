"""``StepContext`` — état d'un pas d'entraînement, passé aux callbacks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_PPL_CEILING = 20.0


@dataclass(frozen=True, slots=True)
class StepContext:
    """Photographie d'un pas, telle que la voient les observateurs.

    Les callbacks reçoivent cet objet plutôt que le trainer entier : ils ne
    peuvent donc pas modifier la boucle d'optimisation par mégarde, et ils
    restent testables sans GPU.

    ``metrics`` est **volontairement mutable** : c'est le tableau de bord partagé
    de l'événement en cours. Un callback qui produit des mesures (scoring) les y
    dépose, un callback qui publie (wandb, console) les y lit. L'ordre de la
    :class:`CallbackList` fait donc partie du contrat — les producteurs avant les
    publieurs.
    """

    step: int
    max_steps: int
    metrics: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    is_main_process: bool = True
    model: Any = None
    """Modèle déballé — nécessaire aux callbacks qui génèrent ou sauvegardent."""

    @property
    def progress(self) -> float:
        return self.step / self.max_steps if self.max_steps else 0.0

    @property
    def is_final(self) -> bool:
        return self.step >= self.max_steps

    def every(self, interval: int) -> bool:
        """Vrai à chaque multiple de ``interval`` (jamais si ``interval <= 0``)."""
        return interval > 0 and self.step > 0 and self.step % interval == 0

    def with_metrics(self, **metrics: float) -> StepContext:
        return StepContext(
            step=self.step,
            max_steps=self.max_steps,
            metrics={**self.metrics, **metrics},
            learning_rate=self.learning_rate,
            grad_norm=self.grad_norm,
            is_main_process=self.is_main_process,
            model=self.model,
        )


def perplexity(loss: float) -> float:
    """exp(loss), plafonnée — le signal direct « apprend-il à émettre ce texte »."""
    return math.exp(min(loss, _PPL_CEILING))
