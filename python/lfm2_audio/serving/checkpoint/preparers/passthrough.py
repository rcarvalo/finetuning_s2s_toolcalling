"""``OmniPassthroughPreparer`` — checkpoint déjà servable, rien à faire."""

from __future__ import annotations

import logging
from pathlib import Path

from lfm2_audio.ds.checkpoint import CheckpointRequest, Layout
from lfm2_audio.serving.checkpoint.preparers.base import CheckpointPreparer

logger = logging.getLogger(__name__)


class OmniPassthroughPreparer(CheckpointPreparer):
    """Déjà au layout Omni et sans adaptateur : le source est servi tel quel."""

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
