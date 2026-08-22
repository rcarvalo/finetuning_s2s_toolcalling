"""Value objects de la résolution de checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Layout(StrEnum):
    """Layout d'un répertoire de checkpoint, déduit de son ``config.json``."""

    OMNI = "omni"
    """Prêt pour vLLM-Omni (``architectures: [Lfm2AudioOmniModel]``)."""

    LIQUID = "liquid"
    """Checkpoint liquid-audio complet (sections lfm/encoder/depthformer/preprocessor)."""

    BACKBONE = "backbone"
    """Backbone texte seul remappé ``Lfm2ForCausalLM`` — servable, mais sans audio."""

    ADAPTER = "adapter"
    """Adaptateur LoRA (``adapter_config.json``) : a besoin d'un modèle de base."""


@dataclass(frozen=True, slots=True)
class CheckpointRequest:
    """Ce que l'utilisateur demande, avant toute résolution.

    ``interleaved_ratio`` (n_text, n_audio) n'est utilisé que lors d'une fusion
    LoRA : il est alors écrit dans le ``config.json`` de l'export, qui devient la
    source unique du ratio pour le serving.
    """

    spec: str | Path
    backend: str
    adapter: str | Path | None = None
    interleaved_ratio: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ResolvedCheckpoint:
    """Répertoire prêt à être chargé par un backend."""

    path: Path
    layout: Layout
    adapter: Path | None = None
    """Adaptateur à fusionner en mémoire — backend liquid uniquement."""

    @property
    def name(self) -> str:
        return self.path.name
