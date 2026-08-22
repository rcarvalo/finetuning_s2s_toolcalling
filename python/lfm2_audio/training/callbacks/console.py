"""``ConsoleCallback`` — trace lisible de la progression."""

from __future__ import annotations

import logging

from lfm2_audio.training.callback import TrainingCallback
from lfm2_audio.training.step_context import StepContext, perplexity

logger = logging.getLogger(__name__)


class ConsoleCallback(TrainingCallback):
    """Affiche loss, perplexité texte, grad norm et lr à intervalle régulier.

    ``text_ppl`` est le signal le plus direct sur le tool calling : c'est la
    perplexité sur le flux texte, là où le modèle apprend à émettre le span
    d'appel. Elle bouge bien avant la loss globale.
    """

    def __init__(self, interval: int = 10) -> None:
        self._interval = interval

    def on_step_end(self, context: StepContext) -> None:
        if not (context.is_main_process and context.every(self._interval)):
            return
        text_loss = context.metrics.get("text_loss")
        suffix = f" text_ppl={perplexity(text_loss):.2f}" if text_loss is not None else ""
        logger.info(
            "step %d/%d loss=%.4f%s grad=%.2f lr=%.2e",
            context.step,
            context.max_steps,
            context.metrics.get("loss", float("nan")),
            suffix,
            context.grad_norm,
            context.learning_rate,
        )

    def on_validate(self, context: StepContext) -> None:
        if not context.is_main_process:
            return
        val_text_loss = context.metrics.get("val_text_loss")
        suffix = f" val_text_ppl={perplexity(val_text_loss):.2f}" if val_text_loss is not None else ""
        logger.info(
            "VAL step %d/%d val_loss=%.4f%s",
            context.step,
            context.max_steps,
            context.metrics.get("val_loss", float("nan")),
            suffix,
        )
