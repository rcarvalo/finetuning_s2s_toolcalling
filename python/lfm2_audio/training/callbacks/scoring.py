"""``ScoringCallback`` — fait tourner les scorers d'éval pendant l'entraînement.

C'est le point qui relie les deux mondes : les métriques suivies tous les N pas
sont **exactement les objets** de la pipeline d'évaluation. Le WER affiché au pas
500 et celui du rapport final sortent du même code, donc ils sont comparables —
ce qui n'est pas le cas quand chaque contexte réimplémente sa métrique.

Ce que ça permet concrètement : voir le WER remonter au pas 800 et arrêter
l'entraînement avant d'avoir cassé les têtes audio, au lieu de le découvrir à
l'évaluation finale.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from lfm2_audio.evaluation.generator import ResponseGenerator
from lfm2_audio.evaluation.pipeline import EvaluationPipeline
from lfm2_audio.evaluation.question_set import QuestionSet
from lfm2_audio.training.callback import TrainingCallback
from lfm2_audio.training.step_context import StepContext

logger = logging.getLogger(__name__)

METRIC_PREFIX = "score"


class ScoringCallback(TrainingCallback):
    """Évalue le modèle en cours d'entraînement sur un petit jeu de questions.

    Le jeu doit rester **petit** : générer de l'audio puis le transcrire coûte
    bien plus qu'un pas d'entraînement. Quelques dizaines de cas suffisent à voir
    une tendance ; la campagne complète reste pour la fin.
    """

    def __init__(
        self,
        questions: QuestionSet,
        pipeline: EvaluationPipeline,
        generator_factory: Callable[[Any], ResponseGenerator] | None = None,
        *,
        interval: int = 500,
        at_start: bool = False,
        generator: ResponseGenerator | None = None,
    ) -> None:
        self._questions = questions
        self._pipeline = pipeline
        self._generator = generator
        self._generator_factory = generator_factory
        self._interval = interval
        self._at_start = at_start

    def on_train_begin(self, context: StepContext) -> None:
        """Mesure de référence : sans elle, aucune des valeurs suivantes ne se lit."""
        if self._at_start and context.is_main_process:
            self._evaluate(context, label="baseline")

    def on_step_end(self, context: StepContext) -> None:
        if context.is_main_process and context.every(self._interval):
            self._evaluate(context, label=f"step-{context.step}")

    def on_train_end(self, context: StepContext) -> None:
        if context.is_main_process:
            self._evaluate(context, label="final")

    # ------------------------------------------------------------------ #

    def _evaluate(self, context: StepContext, *, label: str) -> None:
        generator = self._resolve_generator(context)
        if generator is None:
            logger.warning("aucun générateur de réponses : scoring ignoré (%s)", label)
            return

        logger.info("scoring %s sur %d questions", label, len(self._questions))
        report = self._pipeline.run(self._questions, generator, context={"phase": label, "step": context.step})
        logger.info("\n%s", report.render())

        # Dépôt dans le tableau de bord de l'événement : les callbacks publieurs
        # (wandb, console) les diffusent sans savoir qu'elles viennent d'un scorer.
        context.metrics.update(
            {
                f"{METRIC_PREFIX}/{summary.scorer}": summary.mean
                for summary in report.measured_metrics
                if summary.mean is not None
            }
        )

    def _resolve_generator(self, context: StepContext) -> ResponseGenerator | None:
        """Générateur injecté, ou construit autour du modèle en cours."""
        if self._generator is not None:
            return self._generator
        if self._generator_factory is not None and context.model is not None:
            return self._generator_factory(context.model)
        return None
