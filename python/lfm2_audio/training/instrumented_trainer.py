"""``InstrumentedTrainer`` — wrapper du ``Trainer`` officiel de liquid-audio.

Le ``Trainer`` amont fait le strict nécessaire (boucle, optimiseur, scheduler) et
n'imprime que loss/lr. Cette sous-classe ajoute deux choses, et deux seulement :

1. le **clipping de gradient**, que la boucle de base n'a pas — sans lui,
   l'entraînement peut diverger sur un batch aberrant ;
2. l'**émission d'événements** vers une :class:`CallbackList`.

Tout le reste — journalisation, wandb, sauvegardes, push Hub, scoring périodique
— vit dans des callbacks configurables. La boucle d'optimisation, elle, reste
celle d'amont : c'est ce qui permet de suivre les versions de liquid-audio sans
réécrire l'entraînement.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from liquid_audio.trainer import Trainer

from lfm2_audio.training.callback import TrainingCallback
from lfm2_audio.training.callback_list import CallbackList
from lfm2_audio.training.step_context import StepContext


class InstrumentedTrainer(Trainer):
    """Trainer officiel + clipping + observateurs."""

    def __init__(
        self,
        *,
        callbacks: list[TrainingCallback] | None = None,
        grad_clip: float = 1.0,
        **trainer_kwargs: Any,
    ) -> None:
        super().__init__(**trainer_kwargs)
        self.grad_clip = grad_clip
        self.callbacks = CallbackList(callbacks or [])
        self._grad_norm = 0.0

    # ------------------------------------------------------------------ #
    # Boucle
    # ------------------------------------------------------------------ #

    def train_step(self, batch: Any) -> Any:
        """Un pas, avec capture de la norme de gradient et clipping."""
        self.optimizer.zero_grad()
        batch = batch.to(self.accelerator.device)
        with self.accelerator.autocast():
            out = self.model(batch)
        self.accelerator.backward(out.loss)
        if self.grad_clip:
            self._grad_norm = float(self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip))
        self.optimizer.step()
        self.scheduler.step()
        return out

    def train(self) -> None:
        self.time = time.monotonic()
        self.callbacks.on_train_begin(self._context())

        iterator = iter(self.train_loader)
        while self.step < self.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                self.epoch += 1
                iterator = iter(self.train_loader)
                batch = next(iterator)

            out = self.train_step(batch)
            self.step += 1
            self.callbacks.on_step_end(self._context(self._losses(out)))

            if self.val_loader is not None and self.step % self.val_interval == 0:
                self.model.eval()
                self.callbacks.on_validate(self._context(self.validate()))
                self.model.train()

        self.accelerator.wait_for_everyone()
        self.accelerator.save_model(
            self.accelerator.unwrap_model(self.model),
            f"{self.output_dir}/final",
            max_shard_size="5GB",
            safe_serialization=True,
        )
        self.callbacks.on_train_end(self._context())
        self.callbacks.close()
        self.accelerator.end_training()

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Moyennes de validation, préfixées ``val_``. Vide s'il n'y a pas de loader."""
        if self.val_loader is None:
            return {}

        device = self.accelerator.device
        sums, count = torch.zeros(3, device=device), torch.zeros(1, device=device)
        for raw_batch in self.val_loader:
            batch = raw_batch.to(device)
            with self.accelerator.autocast():
                out = self.model(batch)
            sums += torch.stack(
                [
                    out.loss.detach().reshape(()),
                    out.text_loss.detach().reshape(()),
                    out.audio_loss.detach().reshape(()),
                ]
            )
            count += 1

        sums = self.accelerator.reduce(sums, reduction="sum")
        count = self.accelerator.reduce(count, reduction="sum").clamp_min(1)
        loss, text_loss, audio_loss = (sums / count).tolist()
        return {"val_loss": loss, "val_text_loss": text_loss, "val_audio_loss": audio_loss}

    # ------------------------------------------------------------------ #

    def _losses(self, out: Any) -> dict[str, float]:
        """Pertes du pas, réduites entre processus."""
        reduce = self.accelerator.reduce
        return {
            "loss": reduce(out.loss.detach(), reduction="mean").item(),
            "text_loss": reduce(out.text_loss.detach(), reduction="mean").item(),
            "audio_loss": reduce(out.audio_loss.detach(), reduction="mean").item(),
        }

    def _context(self, metrics: dict[str, float] | None = None) -> StepContext:
        return StepContext(
            step=self.step,
            max_steps=self.max_steps,
            metrics=dict(metrics or {}),
            learning_rate=self.optimizer.param_groups[0]["lr"],
            grad_norm=self._grad_norm,
            is_main_process=self.accelerator.is_main_process,
            model=self.accelerator.unwrap_model(self.model),
        )
