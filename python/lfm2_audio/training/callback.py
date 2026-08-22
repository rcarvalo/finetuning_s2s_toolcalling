"""``TrainingCallback`` — observateur de la boucle d'entraînement.

*Observer* : le trainer ne connaît que ce contrat. Ajouter le suivi d'une
métrique, un push, une sauvegarde ou une éval périodique ne touche donc plus la
boucle — c'est une classe de plus, activée par la config.

Toutes les méthodes ont une implémentation vide : un callback n'écrit que les
moments qui le concernent.
"""

from __future__ import annotations

from lfm2_audio.training.step_context import StepContext


class TrainingCallback:
    """Réagit aux moments clés d'un entraînement."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def on_train_begin(self, context: StepContext) -> None:
        """Avant le premier pas."""

    def on_step_end(self, context: StepContext) -> None:
        """Après chaque pas d'optimisation."""

    def on_validate(self, context: StepContext) -> None:
        """Après une passe de validation (les métriques sont dans le contexte)."""

    def on_train_end(self, context: StepContext) -> None:
        """Après le dernier pas, avant la libération des ressources."""

    def close(self) -> None:
        """Libère les ressources propres au callback. Idempotent."""
