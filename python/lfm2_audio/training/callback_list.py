"""``CallbackList`` — composite de callbacks.

*Composite* : le trainer parle à un seul objet, qui se comporte comme un
callback tout en en pilotant plusieurs. Un callback qui échoue est signalé et
neutralisé pour la suite : perdre le suivi wandb ne doit pas perdre
l'entraînement.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence

from lfm2_audio.training.callback import TrainingCallback
from lfm2_audio.training.step_context import StepContext

logger = logging.getLogger(__name__)


class CallbackList(TrainingCallback):
    """Diffuse chaque événement à une liste de callbacks."""

    def __init__(self, callbacks: Sequence[TrainingCallback] = ()) -> None:
        self._callbacks = list(callbacks)
        self._disabled: set[str] = set()

    def __iter__(self) -> Iterator[TrainingCallback]:
        return iter(self._callbacks)

    def __len__(self) -> int:
        return len(self._callbacks)

    def add(self, callback: TrainingCallback) -> None:
        self._callbacks.append(callback)

    def on_train_begin(self, context: StepContext) -> None:
        self._dispatch("on_train_begin", context)

    def on_step_end(self, context: StepContext) -> None:
        self._dispatch("on_step_end", context)

    def on_validate(self, context: StepContext) -> None:
        self._dispatch("on_validate", context)

    def on_train_end(self, context: StepContext) -> None:
        self._dispatch("on_train_end", context)

    def close(self) -> None:
        for callback in self._callbacks:
            try:
                callback.close()
            except Exception as error:  # la fermeture d'un callback n'en bloque pas d'autres
                logger.warning("fermeture de %s échouée : %s", callback.name, error)

    def _dispatch(self, event: str, context: StepContext) -> None:
        for callback in self._callbacks:
            if callback.name in self._disabled:
                continue
            try:
                getattr(callback, event)(context)
            except Exception as error:  # un observateur ne fait pas tomber l'entraînement
                logger.warning(
                    "%s a échoué sur %s (step %d) : %s — désactivé pour la suite",
                    callback.name,
                    event,
                    context.step,
                    error,
                )
                self._disabled.add(callback.name)
