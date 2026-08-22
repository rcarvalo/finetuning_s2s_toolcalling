"""Politiques de gel des sous-modules de LFM2AudioModel (Phase 2).

Cartographie des sous-modules (cf. ``liquid_audio.model.lfm2_audio``) :

- ``lfm``                : backbone LFM2.5-1.2B (cible LoRA) ;
- ``conformer``,
  ``audio_adapter``      : encodeur FastConformer + projection → à GELER pendant
                           le SFT dialogue (préserve la compréhension audio) ;
- ``audio_embedding``,
  ``depthformer``,
  ``depth_linear``,
  ``depth_embeddings``   : têtes audio (RQ-Transformer) → gel recommandé si la
                           voix vanilla convient (risque de dégradation UTMOS),
                           sinon fine-tuning prudent.

Seuil de décision (méthodologie) : si UTMOS chute > 0,3 vs vanilla → geler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ENCODER_PREFIXES = ("conformer.", "audio_adapter.")
AUDIO_HEAD_PREFIXES = ("audio_embedding.", "depthformer.", "depth_linear.", "depth_embeddings.")
BACKBONE_PREFIX = "lfm."


@dataclass(slots=True)
class FreezePolicy:
    freeze_encoder: bool = True
    freeze_audio_heads: bool = False
    freeze_backbone: bool = False  # True en mode LoRA (seuls les adaptateurs s'entraînent)


def apply_freeze_policy(model, policy: FreezePolicy) -> dict[str, int]:
    """Applique ``requires_grad`` selon la politique ; retourne les comptes de paramètres."""
    counts = {"trainable": 0, "frozen": 0}

    for name, param in model.named_parameters():
        freeze = False
        if (
            (policy.freeze_encoder and name.startswith(ENCODER_PREFIXES))
            or (policy.freeze_audio_heads and name.startswith(AUDIO_HEAD_PREFIXES))
            or (policy.freeze_backbone and name.startswith(BACKBONE_PREFIX) and "lora_" not in name)
        ):
            freeze = True

        param.requires_grad = not freeze
        counts["frozen" if freeze else "trainable"] += param.numel()

    logger.info(
        "freeze policy applied: %.1fM trainable / %.1fM frozen",
        counts["trainable"] / 1e6,
        counts["frozen"] / 1e6,
    )
    return counts
