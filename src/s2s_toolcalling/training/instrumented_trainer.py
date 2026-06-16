"""Trainer instrumenté : wandb + grad norm/clip + perplexité + push HF régulier.

Sous-classe le ``Trainer`` officiel de liquid-audio (minimal : il n'imprime que
loss/lr). On ajoute, SANS toucher la boucle d'optimisation :

- **wandb** (loss, text_loss/audio_loss décomposés, **text_perplexity = exp(text_loss)**
  = le signal direct « apprend-il à émettre le texte du tool call », grad_norm, lr) ;
- **clipping** de gradient (le Trainer de base n'en fait pas → instabilité possible) ;
- **validation** enrichie (val_loss + val_text_perplexity) ;
- **push HF** de l'adaptateur LoRA tous les ``push_interval`` steps (+ à la fin).

Import paresseux côté appelant (nécessite ``liquid_audio`` + ``wandb``, GPU).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from liquid_audio.trainer import Trainer

from s2s_toolcalling.training.lora import LoraSettings, save_lora


def _ppl(loss: float) -> float:
    return math.exp(min(loss, 20.0))  # garde-fou overflow


class InstrumentedTrainer(Trainer):
    def __init__(
        self,
        *,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        wandb_config: dict | None = None,
        hub_repo: str | None = None,
        push_interval: int = 0,
        grad_clip: float = 1.0,
        lora_settings: LoraSettings | None = None,
        **trainer_kwargs,
    ) -> None:
        super().__init__(**trainer_kwargs)
        self.grad_clip = grad_clip
        self.hub_repo = hub_repo
        self.push_interval = push_interval
        self.lora_settings = lora_settings
        self._grad_norm = 0.0
        self._wandb = None
        if wandb_project and self.accelerator.is_main_process:
            import wandb

            self._wandb = wandb
            wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config or {})

    # -- pas de gradient → grad norm capturé + clipping (le base n'en a pas) -----
    def train_step(self, batch):
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

    def log(self, out) -> None:
        if not (self.step > 0 and self.step % self.logging_interval == 0):
            return
        red = self.accelerator.reduce
        loss = red(out.loss.detach(), reduction="mean").item()
        text_loss = red(out.text_loss.detach(), reduction="mean").item()
        audio_loss = red(out.audio_loss.detach(), reduction="mean").item()
        lr = self.optimizer.param_groups[0]["lr"]
        self.accelerator.print(
            f"step {self.step}/{self.max_steps} loss={loss:.4f} text_loss={text_loss:.4f} "
            f"text_ppl={_ppl(text_loss):.2f} grad={self._grad_norm:.2f} lr={lr:.2e}"
        )
        if self._wandb:
            self._wandb.log(
                {"train/loss": loss, "train/text_loss": text_loss, "train/audio_loss": audio_loss,
                 "train/text_ppl": _ppl(text_loss), "train/grad_norm": self._grad_norm, "train/lr": lr},
                step=self.step,
            )

    @torch.no_grad()
    def validate(self) -> None:
        if self.val_loader is None:
            return
        dev = self.accelerator.device
        sums, count = torch.zeros(3, device=dev), torch.zeros(1, device=dev)
        for batch in self.val_loader:
            batch = batch.to(dev)
            with self.accelerator.autocast():
                out = self.model(batch)
            sums += torch.stack([out.loss.detach().reshape(()), out.text_loss.detach().reshape(()),
                                 out.audio_loss.detach().reshape(())])
            count += 1
        sums = self.accelerator.reduce(sums, reduction="sum")
        count = self.accelerator.reduce(count, reduction="sum").clamp_min(1)
        loss, text_loss, audio_loss = (sums / count).tolist()
        self.accelerator.print(
            f"VAL step {self.step}/{self.max_steps} val_loss={loss:.4f} val_text_ppl={_ppl(text_loss):.2f}"
        )
        if self._wandb:
            self._wandb.log(
                {"val/loss": loss, "val/text_loss": text_loss, "val/audio_loss": audio_loss,
                 "val/text_ppl": _ppl(text_loss)},
                step=self.step,
            )

    def _push_adapter(self, label: str) -> None:
        if not (self.hub_repo and self.lora_settings and self.accelerator.is_main_process):
            return
        from huggingface_hub import HfApi

        out = Path(self.output_dir) / "adapter"
        save_lora(self.accelerator.unwrap_model(self.model), out, self.lora_settings)
        api = HfApi()
        api.create_repo(self.hub_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=self.hub_repo,
                          commit_message=f"adapter @ {label} (step {self.step})")
        self.accelerator.print(f"pushed adapter -> {self.hub_repo} ({label})")

    def train(self) -> None:
        self.time = time.monotonic()
        self.accelerator.print("Start training")
        it = iter(self.train_loader)
        while self.step < self.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                self.epoch += 1
                it = iter(self.train_loader)
                batch = next(it)
            out = self.train_step(batch)
            self.step += 1
            self.log(out)
            if self.step % self.save_interval == 0:
                self.accelerator.save_state()
            if self.push_interval and self.step % self.push_interval == 0:
                self._push_adapter(f"step-{self.step}")
            if self.val_loader is not None and self.step % self.val_interval == 0:
                self.model.eval()
                self.validate()
                self.model.train()
        self.accelerator.wait_for_everyone()
        self.accelerator.save_model(self.accelerator.unwrap_model(self.model),
                                    f"{self.output_dir}/final", max_shard_size="5GB", safe_serialization=True)
        self._push_adapter("final")
        self.accelerator.end_training()
        if self._wandb:
            self._wandb.finish()
