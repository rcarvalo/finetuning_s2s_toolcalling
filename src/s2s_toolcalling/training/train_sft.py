"""Lanceur SFT Phase 2 — réutilise le ``Trainer`` officiel de liquid-audio.

On ne réimplémente PAS la boucle d'entraînement : ``liquid_audio.trainer.Trainer``
gère déjà Accelerate (bf16, DDP find_unused_parameters), le scheduler
warmup+cosine, la validation et les checkpoints. Ce module ajoute uniquement :

1. l'injection LoRA dans le backbone + politiques de gel (encodeur, têtes
   audio) appliquées au modèle **avant** que le Trainer ne crée l'optimiseur —
   via un hook sur ``from_pretrained`` (les paramètres gelés ne reçoivent
   jamais de gradient, AdamW les ignore donc sans modification du Trainer) ;
2. la sauvegarde de l'adaptateur LoRA seul en fin d'entraînement (en plus du
   ``final/`` complet écrit par le Trainer).

Usage (config YAML, cf. ``configs/``) :

    accelerate launch -m s2s_toolcalling.training.train_sft \\
        --config configs/phase2b_sft_toolcalling.yaml

Phase 2a (adaptation FR, mode sequential) et 2b (SFT interleaved + tool calls)
utilisent le même lanceur — seuls le dataset et les politiques de gel changent.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from s2s_toolcalling.training.freeze import FreezePolicy
from s2s_toolcalling.training.lora import LoraSettings


@dataclass(slots=True)
class TrainConfig:
    """Hyperparamètres : ceux du Trainer officiel + LoRA/gel.

    Défauts orientés LoRA mono-GPU (lr 1e-4) ; descendre vers 2e-5 si
    ``lora.enabled: false`` (full fine-tuning).
    """

    model_id: str = "LiquidAI/LFM2.5-Audio-1.5B"
    train_dataset: str = ""
    val_dataset: str | None = None
    context_length: int = 4096

    lr: float = 1e-4
    weight_decay: float = 0.1
    min_lr_ratio: float = 0.1
    max_steps: int = 2000
    warmup_steps: int = 100
    batch_size: int = 2
    dataloader_num_workers: int = 2

    logging_interval: int = 10
    save_interval: int = 500
    val_interval: int = 200
    output_dir: str = "outputs/sft"

    # instrumentation (InstrumentedTrainer) — tout optionnel
    grad_clip: float = 1.0
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    hub_repo: str | None = None          # ex. Rcarvalo/lfm25-tc-en-adapter
    push_interval: int = 0               # push HF de l'adaptateur tous les N steps (0 = jamais)

    lora: LoraSettings = field(default_factory=LoraSettings)
    freeze: FreezePolicy = field(default_factory=FreezePolicy)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        lora = LoraSettings(**raw.pop("lora", {}))
        freeze = FreezePolicy(**raw.pop("freeze", {}))
        return cls(**raw, lora=lora, freeze=freeze)


@contextmanager
def _model_construction_hook(cfg: TrainConfig):
    """Intercepte ``LFM2AudioModel.from_pretrained`` dans le module trainer pour
    injecter LoRA + gel entre le chargement du modèle et la création de
    l'optimiseur (seul point d'accroche offert par le Trainer officiel)."""
    import liquid_audio.trainer as trainer_module
    from liquid_audio import LFM2AudioModel

    from s2s_toolcalling.training.freeze import apply_freeze_policy
    from s2s_toolcalling.training.lora import inject_lora

    class _ModelFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model = LFM2AudioModel.from_pretrained(*args, **kwargs)
            if cfg.lora.enabled:
                inject_lora(model, cfg.lora)
                cfg.freeze.freeze_backbone = True  # seuls les adaptateurs du backbone s'entraînent
            counts = apply_freeze_policy(model, cfg.freeze)
            print(f"trainable: {counts['trainable'] / 1e6:.1f}M params / frozen: {counts['frozen'] / 1e6:.1f}M")
            return model

    original = trainer_module.LFM2AudioModel
    trainer_module.LFM2AudioModel = _ModelFactory  # type: ignore[assignment]
    try:
        yield
    finally:
        trainer_module.LFM2AudioModel = original


def build_trainer(cfg: TrainConfig):
    from liquid_audio.data.dataloader import LFM2DataLoader

    from s2s_toolcalling.training.instrumented_trainer import InstrumentedTrainer

    train_data = LFM2DataLoader(cfg.train_dataset, context_length=cfg.context_length)
    val_data = LFM2DataLoader(cfg.val_dataset, context_length=cfg.context_length) if cfg.val_dataset else None

    with _model_construction_hook(cfg):
        return InstrumentedTrainer(
            # instrumentation
            wandb_project=cfg.wandb_project,
            wandb_run_name=cfg.wandb_run_name,
            wandb_config=asdict(cfg),
            hub_repo=cfg.hub_repo,
            push_interval=cfg.push_interval,
            grad_clip=cfg.grad_clip,
            lora_settings=cfg.lora if cfg.lora.enabled else None,
            # hyperparams du Trainer officiel
            model_id=cfg.model_id,
            train_data=train_data,
            val_data=val_data,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            min_ratio=cfg.min_lr_ratio,
            max_steps=cfg.max_steps,
            warmup_steps=cfg.warmup_steps,
            batch_size=cfg.batch_size,
            dataloader_num_workers=cfg.dataloader_num_workers,
            logging_interval=cfg.logging_interval,
            save_interval=cfg.save_interval,
            val_interval=cfg.val_interval,
            output_dir=cfg.output_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    print("Config:", asdict(cfg))

    trainer = build_trainer(cfg)
    trainer.train()

    # Le Trainer écrit déjà `<output_dir>/final` (modèle complet, LoRA non fusionné) ;
    # on ajoute l'adaptateur seul, léger à versionner et à charger via lora.load_lora.
    if cfg.lora.enabled and trainer.accelerator.is_main_process:
        from s2s_toolcalling.training.lora import save_lora

        path = save_lora(trainer.accelerator.unwrap_model(trainer.model), Path(cfg.output_dir) / "final_adapter", cfg.lora)
        print(f"LoRA adapter saved to {path}")


if __name__ == "__main__":
    main()
