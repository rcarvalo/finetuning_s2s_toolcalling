"""``CheckpointPreparer`` — stratégie d'amenée au layout vLLM-Omni.

Chaque stratégie couvre UN cas de figure. Le résolveur choisit la première
applicable ; ajouter un cas (quantisation, distillation…) revient à ajouter une
classe, sans toucher au résolveur.

Toutes écrivent dans un répertoire de cache et n'y posent le marqueur de fin
qu'après succès : un run interrompu ne sera jamais réutilisé tel quel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from lfm2_audio.ds.checkpoint import CheckpointRequest, Layout

READY_MARKER = ".lfm2_ready"


class CheckpointPreparer(ABC):
    """Amène un checkpoint source au layout Omni."""

    @abstractmethod
    def handles(self, layout: Layout, *, has_adapter: bool) -> bool:
        """Vrai si cette stratégie couvre ce couple (layout, présence d'adaptateur).

        Doit rester **bon marché** : le résolveur l'interroge avant toute
        décision, et un préparateur paresseux y répond sans charger son module.
        """

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
    def mark_ready(target: Path) -> None:
        """Pose le marqueur de complétude — après succès, jamais avant."""
        (target / READY_MARKER).write_text("", encoding="utf-8")
