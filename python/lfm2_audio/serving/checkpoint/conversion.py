"""Conversion d'un checkpoint liquid-audio vers le layout vLLM-Omni.

Réécriture du ``config.json`` uniquement : les poids d'un export complet sont
déjà au bon format.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lfm2_audio.vllm_plugin.constants import AUDIO_EOA_PLACEHOLDER_ID, AUDIO_FRAME_PLACEHOLDER_ID
from lfm2_audio.vllm_plugin.convert_checkpoint import convert

logger = logging.getLogger(__name__)


def convert_to_omni(source: Path, target: Path) -> None:
    """Écrit dans ``target`` la version Omni du checkpoint ``source``."""
    logger.info("conversion vers le layout vLLM-Omni : %s", target)
    convert(source, target, frame_id=AUDIO_FRAME_PLACEHOLDER_ID, eoa_id=AUDIO_EOA_PLACEHOLDER_ID)
