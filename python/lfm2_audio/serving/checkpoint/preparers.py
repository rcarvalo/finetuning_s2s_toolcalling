"""Préparateurs de checkpoint — *Strategy*.

Chaque stratégie sait amener UN cas de figure au layout attendu par vLLM-Omni.
Le résolveur choisit la première applicable ; ajouter un cas (quantisation,
distillation…) = ajouter une classe.

Toutes écrivent dans un répertoire de cache et n'y posent le marqueur de fin
qu'après succès : un run interrompu ne sera jamais réutilisé tel quel.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from lfm2_audio.ds.checkpoint import CheckpointRequest, Layout

logger = logging.getLogger(__name__)

READY_MARKER = ".lfm2_ready"


class CheckpointPreparer(ABC):
    """Amène un checkpoint source au layout Omni."""

    @abstractmethod
    def handles(self, layout: Layout, *, has_adapter: bool) -> bool:
        """Vrai si cette stratégie couvre ce couple (layout, présence d'adaptateur)."""

    @abstractmethod
    def prepare(
        self,
        source: Path,
        target: Path,
        request: CheckpointRequest,
        adapter: Path | None,
    ) -> Path:
        """Produit le répertoire Omni et retourne son chemin."""

    @property
    def name(self) -> str:
        return type(self).__name__

    @staticmethod
    def _mark_ready(target: Path) -> None:
        (target / READY_MARKER).write_text("", encoding="utf-8")


class OmniPassthroughPreparer(CheckpointPreparer):
    """Déjà au layout Omni et sans adaptateur : rien à faire."""

    def handles(self, layout: Layout, *, has_adapter: bool) -> bool:
        return layout is Layout.OMNI and not has_adapter

    def prepare(
        self,
        source: Path,
        target: Path,
        request: CheckpointRequest,
        adapter: Path | None,
    ) -> Path:
        logger.info("checkpoint déjà au layout vLLM-Omni : %s", source)
        return source


class LiquidConversionPreparer(CheckpointPreparer):
    """Layout liquid-audio sans adaptateur : réécriture du ``config.json``."""

    def handles(self, layout: Layout, *, has_adapter: bool) -> bool:
        return layout is Layout.LIQUID and not has_adapter

    def prepare(
        self,
        source: Path,
        target: Path,
        request: CheckpointRequest,
        adapter: Path | None,
    ) -> Path:
        _convert_to_omni(source, target)
        self._mark_ready(target)
        return target


class LoraMergePreparer(CheckpointPreparer):
    """Base + adaptateur LoRA : fusion (GPU) puis conversion.

    C'est la seule stratégie coûteuse — d'où le cache : elle ne tourne qu'une
    fois par couple (base, adaptateur, ratio).
    """

    def handles(self, layout: Layout, *, has_adapter: bool) -> bool:
        return layout in (Layout.LIQUID, Layout.OMNI) and has_adapter

    def prepare(
        self,
        source: Path,
        target: Path,
        request: CheckpointRequest,
        adapter: Path | None,
    ) -> Path:
        from lfm2_audio.training.export_checkpoint import export_full

        merged = target.with_name(target.name + "-merged")
        merged.mkdir(parents=True, exist_ok=True)
        n_text, n_audio = request.interleaved_ratio or (None, None)

        logger.info("fusion de l'adaptateur %s dans %s", adapter, source)
        export_full(str(source), str(adapter), merged, "cuda", n_text, n_audio)

        _convert_to_omni(merged, target)
        self._mark_ready(target)
        return target


def _convert_to_omni(source: Path, target: Path) -> None:
    """Réécrit le ``config.json`` au layout Omni (les poids sont déjà bons)."""
    from lfm2_audio.vllm_plugin.constants import (
        AUDIO_EOA_PLACEHOLDER_ID,
        AUDIO_FRAME_PLACEHOLDER_ID,
    )
    from lfm2_audio.vllm_plugin.convert_checkpoint import convert

    logger.info("conversion vers le layout vLLM-Omni : %s", target)
    convert(
        source,
        target,
        frame_id=AUDIO_FRAME_PLACEHOLDER_ID,
        eoa_id=AUDIO_EOA_PLACEHOLDER_ID,
    )


DEFAULT_PREPARERS: tuple[CheckpointPreparer, ...] = (
    OmniPassthroughPreparer(),
    LoraMergePreparer(),
    LiquidConversionPreparer(),
)
