"""Stratégies de préparation d'un checkpoint, une par fichier.

``DEFAULT_PREPARERS`` fixe l'ordre d'essai. La fusion LoRA y figure derrière un
:class:`LazyPreparer` : elle se déclare applicable sans charger peft ni torch, et
n'est résolue que si un adaptateur est effectivement fourni.
"""

from lfm2_audio.core.lazy_component import LazyComponent
from lfm2_audio.ds.checkpoint import Layout
from lfm2_audio.serving.checkpoint.preparers.base import READY_MARKER, CheckpointPreparer
from lfm2_audio.serving.checkpoint.preparers.lazy import LazyPreparer
from lfm2_audio.serving.checkpoint.preparers.liquid_conversion import LiquidConversionPreparer
from lfm2_audio.serving.checkpoint.preparers.passthrough import OmniPassthroughPreparer

LORA_MERGE = LazyComponent(
    module="lfm2_audio.serving.checkpoint.preparers.lora_merge",
    class_name="LoraMergePreparer",
    requires=("torch", "peft", "liquid_audio"),
    extra="train",
)

DEFAULT_PREPARERS: tuple[CheckpointPreparer, ...] = (
    OmniPassthroughPreparer(),
    LazyPreparer(
        LORA_MERGE,
        lambda layout, has_adapter: layout in (Layout.LIQUID, Layout.OMNI) and has_adapter,
        label="LoraMergePreparer",
    ),
    LiquidConversionPreparer(),
)

__all__ = [
    "DEFAULT_PREPARERS",
    "LORA_MERGE",
    "READY_MARKER",
    "CheckpointPreparer",
    "LazyPreparer",
    "LiquidConversionPreparer",
    "OmniPassthroughPreparer",
]
