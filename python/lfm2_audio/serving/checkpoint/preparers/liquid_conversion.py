"""``LiquidConversionPreparer`` — layout liquid-audio à convertir."""

from __future__ import annotations

from pathlib import Path

from lfm2_audio.ds.checkpoint import CheckpointRequest, Layout
from lfm2_audio.serving.checkpoint.conversion import convert_to_omni
from lfm2_audio.serving.checkpoint.preparers.base import CheckpointPreparer


class LiquidConversionPreparer(CheckpointPreparer):
    """Réécrit le ``config.json`` au layout Omni. Les poids sont déjà bons."""

    def handles(self, layout: Layout, *, has_adapter: bool) -> bool:
        return layout is Layout.LIQUID and not has_adapter

    def prepare(
        self,
        source: Path,
        target: Path,
        request: CheckpointRequest,
        adapter: Path | None,
    ) -> Path:
        convert_to_omni(source, target)
        self.mark_ready(target)
        return target
