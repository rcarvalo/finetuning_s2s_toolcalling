"""``CheckpointCallback`` — sauvegarde périodique de l'état d'entraînement."""

from __future__ import annotations

import logging
from typing import Any

from lfm2_audio.training.callback import TrainingCallback
from lfm2_audio.training.step_context import StepContext

logger = logging.getLogger(__name__)


class CheckpointCallback(TrainingCallback):
    """Délègue à ``accelerator.save_state`` tous les ``interval`` pas.

    L'accelerator est injecté plutôt que lu sur le contexte : la sauvegarde est
    collective (tous les rangs y participent), c'est le seul callback qui ne
    doit surtout pas se restreindre au processus principal.
    """

    def __init__(self, accelerator: Any, *, interval: int = 500) -> None:
        self._accelerator = accelerator
        self._interval = interval

    def on_step_end(self, context: StepContext) -> None:
        if context.every(self._interval):
            logger.info("sauvegarde de l'état au step %d", context.step)
            self._accelerator.save_state()
