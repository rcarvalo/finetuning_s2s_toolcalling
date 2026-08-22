"""How many optimizer steps a recipe asks for.

Kept out of :mod:`lfm2_audio.training.train_sft`: that module imports torch and
liquid-audio at load time, and this policy is pure — it must stay testable on a
machine with neither.
"""

from __future__ import annotations

import logging
from typing import Any

from lfm2_audio.core.errors import TrainingConfigError
from lfm2_audio.ds.training_config import TrainingConfig

logger = logging.getLogger(__name__)


def resolve_max_steps(config: TrainingConfig, train_data: Any) -> int:
    """Steps to run: derived from ``num_epochs`` when set, else ``max_steps``.

    A recipe reasons in passes over the data; the trainer only counts steps, and
    the corpus size is not known until the loader exists. When the loader cannot
    report a length, say so instead of guessing a step count.
    """
    if not config.num_epochs:
        return config.max_steps
    try:
        size = len(train_data)
    except TypeError as exc:
        message = (
            f"{type(train_data).__name__} does not expose a length, so `num_epochs` "
            "cannot be converted into steps. Set `max_steps` in the recipe instead."
        )
        raise TrainingConfigError(message) from exc
    steps = config.steps_for(size)
    logger.info("%s epoch(s) over %d examples -> %d steps", config.num_epochs, size, steps)
    return steps
