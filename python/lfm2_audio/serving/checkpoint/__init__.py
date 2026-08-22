"""Résolution d'un checkpoint : spécification utilisateur → répertoire servable.

Décomposé en collaborateurs à responsabilité unique :
sources (*chain of responsibility*) → détection de layout → préparation
(*strategy*), le tout orchestré par :class:`CheckpointResolver`.
"""

from lfm2_audio.serving.checkpoint.detector import LayoutDetector
from lfm2_audio.serving.checkpoint.preparers import CheckpointPreparer
from lfm2_audio.serving.checkpoint.resolver import CheckpointResolver
from lfm2_audio.serving.checkpoint.sources import CheckpointSource, SourceChain

__all__ = [
    "CheckpointPreparer",
    "CheckpointResolver",
    "CheckpointSource",
    "LayoutDetector",
    "SourceChain",
]
