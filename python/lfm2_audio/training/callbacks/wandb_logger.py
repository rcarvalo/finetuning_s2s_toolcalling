"""``WandbCallback`` — suivi des métriques dans Weights & Biases."""

from __future__ import annotations

import logging
from typing import Any

import wandb

from lfm2_audio.training.callback import TrainingCallback
from lfm2_audio.training.step_context import StepContext, perplexity

logger = logging.getLogger(__name__)


class WandbCallback(TrainingCallback):
    """Pousse toutes les métriques du contexte, préfixées par phase.

    Les métriques de scoring (WER, tool call…) arrivent par le même chemin que
    la loss : un scorer ajouté à la config apparaît dans wandb sans ligne de
    code supplémentaire.
    """

    def __init__(
        self,
        project: str,
        *,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._project = project
        self._run_name = run_name
        self._config = config or {}
        self._run: Any = None

    def on_train_begin(self, context: StepContext) -> None:
        if not context.is_main_process:
            return
        self._run = wandb.init(project=self._project, name=self._run_name, config=self._config)

    def on_step_end(self, context: StepContext) -> None:
        self._log(context, prefix="train")

    def on_validate(self, context: StepContext) -> None:
        self._log(context, prefix="val")

    def _log(self, context: StepContext, *, prefix: str) -> None:
        if self._run is None or not context.metrics:
            return
        payload: dict[str, float] = {}
        for key, value in context.metrics.items():
            # Les clés déjà préfixées (val_*, score/*) gardent leur espace de noms.
            name = key if "/" in key or key.startswith("val_") else f"{prefix}/{key}"
            payload[name] = value
            if key.endswith("text_loss"):
                payload[f"{name}_ppl"] = perplexity(value)
        payload[f"{prefix}/lr"] = context.learning_rate
        payload[f"{prefix}/grad_norm"] = context.grad_norm
        self._run.log(payload, step=context.step)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
