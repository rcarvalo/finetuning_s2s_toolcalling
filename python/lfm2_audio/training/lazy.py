"""Composants d'entraînement résolus par chemin d'import.

Même mécanisme que pour les scorers : les modules concrets portent leurs imports
lourds (wandb, huggingface_hub) en tête, et ne sont chargés que si la config les
active. Lire une recette d'entraînement ne doit rien installer.
"""

from __future__ import annotations

from lfm2_audio.scorer.lazy import LazyComponent

WANDB_CALLBACK = LazyComponent(
    module="lfm2_audio.training.callbacks.wandb_logger",
    class_name="WandbCallback",
    requires=("wandb",),
    extra="train",
)

CHECKPOINT_CALLBACK = LazyComponent(
    module="lfm2_audio.training.callbacks.checkpoint",
    class_name="CheckpointCallback",
    extra="train",
)

HUB_PUSH_CALLBACK = LazyComponent(
    module="lfm2_audio.training.callbacks.hub_push",
    class_name="HubPushCallback",
    requires=("huggingface_hub",),
    extra="train",
)
