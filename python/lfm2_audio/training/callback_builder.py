"""``CallbackBuilder`` — assemble les observateurs d'un entraînement depuis la config.

*Builder* : le lanceur ne décide plus quoi instrumenter ; il lit une recette. Un
projet qui veut suivre le DNSMOS tous les 200 pas écrit trois lignes de YAML, pas
une ligne de Python.

L'**ordre** compte et n'est pas cosmétique : les callbacks qui *produisent* des
métriques (scoring) passent avant ceux qui les *publient* (console, wandb), afin
que les mesures du pas courant soient déjà dans le tableau de bord partagé.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from lfm2_audio.ds.training_config import TrainingConfig
from lfm2_audio.evaluation.pipeline import EvaluationPipeline
from lfm2_audio.evaluation.question_set import QuestionSet
from lfm2_audio.scorer.factory import ScorerFactory
from lfm2_audio.training.callback import TrainingCallback
from lfm2_audio.training.callbacks.console import ConsoleCallback
from lfm2_audio.training.callbacks.scoring import ScoringCallback
from lfm2_audio.training.lazy import CHECKPOINT_CALLBACK, HUB_PUSH_CALLBACK, WANDB_CALLBACK
from lfm2_audio.training.lora import LoraSettings

logger = logging.getLogger(__name__)


class CallbackBuilder:
    """Construit la liste d'observateurs décrite par une :class:`TrainingConfig`."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        accelerator: Any = None,
        generator_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self._config = config
        self._accelerator = accelerator
        self._generator_factory = generator_factory

    def build(self) -> list[TrainingCallback]:
        """Producteurs de métriques d'abord, publieurs ensuite."""
        callbacks: list[TrainingCallback] = []

        scoring = self._scoring_callback()
        if scoring is not None:
            callbacks.append(scoring)

        callbacks.append(ConsoleCallback(interval=self._config.logging_interval))
        callbacks.extend(self._publishers())
        callbacks.extend(self._persisters())
        return callbacks

    # ------------------------------------------------------------------ #

    def _scoring_callback(self) -> TrainingCallback | None:
        schedule = self._config.evaluation
        if not schedule.enabled:
            return None
        if not schedule.question_set:
            logger.warning("evaluation.enabled sans question_set : scoring désactivé")
            return None

        questions = QuestionSet.from_jsonl(schedule.question_set, audio_root=schedule.audio_root).take(
            schedule.max_questions
        )
        scorers = ScorerFactory(schedule.scoring).build_all()
        logger.info(
            "suivi métrique : %d questions, scorers %s",
            len(questions),
            [s.name for s in scorers],
        )
        return ScoringCallback(
            questions,
            EvaluationPipeline(scorers),
            self._generator_factory,
            interval=schedule.interval,
            at_start=schedule.at_start,
        )

    def _publishers(self) -> list[TrainingCallback]:
        if not self._config.wandb_project:
            return []
        reason = WANDB_CALLBACK.unavailable_reason()
        if reason is not None:
            logger.warning("suivi wandb désactivé : %s", reason)
            return []
        return [
            WANDB_CALLBACK.build(
                project=self._config.wandb_project,
                run_name=self._config.wandb_run_name,
                config=self._config.as_dict(),
            )
        ]

    def _persisters(self) -> list[TrainingCallback]:
        callbacks: list[TrainingCallback] = []

        if self._accelerator is not None:
            callbacks.append(
                CHECKPOINT_CALLBACK.build(accelerator=self._accelerator, interval=self._config.save_interval)
            )

        if self._config.hub_repo and self._config.lora.enabled:
            reason = HUB_PUSH_CALLBACK.unavailable_reason()
            if reason is not None:
                logger.warning("push Hub désactivé : %s", reason)
                return callbacks
            callbacks.append(
                HUB_PUSH_CALLBACK.build(
                    repo_id=self._config.hub_repo,
                    output_dir=self._config.output_dir,
                    lora_settings=LoraSettings(
                        r=self._config.lora.r,
                        alpha=self._config.lora.alpha,
                        dropout=self._config.lora.dropout,
                    ),
                    interval=self._config.push_interval,
                )
            )
        return callbacks
