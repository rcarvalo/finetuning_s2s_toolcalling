"""Lanceur SFT — assemble le ``Trainer`` officiel, LoRA/gel, et les observateurs.

On ne réimplémente **pas** la boucle d'entraînement :
``liquid_audio.trainer.Trainer`` gère déjà Accelerate (bf16, DDP), le scheduler
warmup+cosine, la validation et les checkpoints. Ce module ajoute :

1. l'injection LoRA + les politiques de gel, appliquées au modèle **avant** que
   le Trainer ne crée l'optimiseur — via un hook sur ``from_pretrained``, seul
   point d'accroche offert par le Trainer amont ;
2. les callbacks décrits par la config (journalisation, wandb, sauvegardes, push
   Hub, et le **scoring périodique** avec les métriques de la pipeline d'éval).

    accelerate launch -m lfm2_audio.cli.train.sft \\
        --config configs/training/phase_en_toolcalling.yaml
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import liquid_audio.trainer as trainer_module
from liquid_audio import LFM2AudioModel
from liquid_audio.data.dataloader import LFM2DataLoader

from lfm2_audio.ds.training_config import TrainingConfig
from lfm2_audio.training.callback_builder import CallbackBuilder
from lfm2_audio.training.freeze import FreezePolicy, apply_freeze_policy
from lfm2_audio.training.instrumented_trainer import InstrumentedTrainer
from lfm2_audio.training.lora import inject_lora
from lfm2_audio.training.lora_settings import LoraSettings

logger = logging.getLogger(__name__)


def lora_settings(config: TrainingConfig) -> LoraSettings:
    """Traduit la section LoRA de la config en réglages d'injection."""
    return LoraSettings(
        enabled=config.lora.enabled,
        r=config.lora.r,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
    )


def freeze_policy(config: TrainingConfig) -> FreezePolicy:
    """Politique de gel. Avec LoRA, le backbone est gelé : seuls les adaptateurs
    s'entraînent, et AdamW ignore les paramètres sans gradient."""
    return FreezePolicy(
        freeze_encoder=config.freeze.freeze_encoder,
        freeze_audio_heads=config.freeze.freeze_audio_heads,
        freeze_backbone=config.freeze.freeze_backbone or config.lora.enabled,
    )


@contextmanager
def model_construction_hook(config: TrainingConfig) -> Iterator[None]:
    """Intercepte ``LFM2AudioModel.from_pretrained`` le temps de la construction.

    C'est l'unique fenêtre entre le chargement des poids et la création de
    l'optimiseur — donc le seul moment où injecter LoRA et geler des modules a
    l'effet attendu sur les groupes de paramètres.
    """
    settings = lora_settings(config)
    policy = freeze_policy(config)

    class _ModelFactory:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            model = LFM2AudioModel.from_pretrained(*args, **kwargs)
            if settings.enabled:
                inject_lora(model, settings)
            counts = apply_freeze_policy(model, policy)
            logger.info(
                "entraînables : %.1fM / gelés : %.1fM",
                counts["trainable"] / 1e6,
                counts["frozen"] / 1e6,
            )
            return model

    original = trainer_module.LFM2AudioModel
    trainer_module.LFM2AudioModel = _ModelFactory
    try:
        yield
    finally:
        trainer_module.LFM2AudioModel = original


def build_trainer(config: TrainingConfig, *, generator_factory: Any = None) -> InstrumentedTrainer:
    """Trainer prêt à tourner, observateurs montés depuis la config."""
    train_data = LFM2DataLoader(config.train_dataset, context_length=config.context_length)
    val_data = LFM2DataLoader(config.val_dataset, context_length=config.context_length) if config.val_dataset else None

    with model_construction_hook(config):
        trainer = InstrumentedTrainer(
            grad_clip=config.grad_clip,
            model_id=config.model_id,
            train_data=train_data,
            val_data=val_data,
            lr=config.lr,
            weight_decay=config.weight_decay,
            min_ratio=config.min_lr_ratio,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            batch_size=config.batch_size,
            dataloader_num_workers=config.dataloader_num_workers,
            logging_interval=config.logging_interval,
            save_interval=config.save_interval,
            val_interval=config.val_interval,
            output_dir=config.output_dir,
        )

    # L'accelerator n'existe qu'une fois le Trainer construit : les callbacks qui
    # en dépendent (sauvegarde d'état) sont donc montés après.
    builder = CallbackBuilder(config, accelerator=trainer.accelerator, generator_factory=generator_factory)
    for callback in builder.build():
        trainer.callbacks.add(callback)
    return trainer
