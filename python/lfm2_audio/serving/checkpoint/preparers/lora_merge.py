"""``LoraMergePreparer`` — fusionne un adaptateur LoRA, puis convertit.

Seule stratégie coûteuse (elle charge le modèle sur GPU) et seule à dépendre de
``training``, donc de peft et torch — d'où son importation en tête et sa
résolution paresseuse par :class:`LazyPreparer`. Servir un checkpoint déjà
converti n'exige ainsi aucune dépendance d'entraînement.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lfm2_audio.ds.checkpoint import CheckpointRequest, Layout
from lfm2_audio.serving.checkpoint.conversion import convert_to_omni
from lfm2_audio.serving.checkpoint.preparers.base import CheckpointPreparer
from lfm2_audio.training.export_checkpoint import export_full

logger = logging.getLogger(__name__)


class LoraMergePreparer(CheckpointPreparer):
    """Base + adaptateur : fusion des poids puis conversion.

    Le cache du résolveur fait qu'elle ne tourne qu'une fois par couple
    (base, adaptateur, ratio interleaved).
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
        merged = target.with_name(target.name + "-merged")
        merged.mkdir(parents=True, exist_ok=True)
        n_text, n_audio = request.interleaved_ratio or (None, None)

        logger.info("fusion de l'adaptateur %s dans %s", adapter, source)
        export_full(str(source), str(adapter), merged, "cuda", n_text, n_audio)

        convert_to_omni(merged, target)
        self.mark_ready(target)
        return target
