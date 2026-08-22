"""Chargement du modèle et backends d'inférence.

Point d'entrée : :class:`lfm2_audio.serving.model.LFM2Audio`, dont la fabrique
``from_pretrained`` résout le checkpoint puis instancie le backend enregistré.
"""

from lfm2_audio.serving.model import LFM2Audio
from lfm2_audio.serving.registry import BACKENDS, BackendRegistry, BackendSpec

__all__ = ["BACKENDS", "BackendRegistry", "BackendSpec", "LFM2Audio"]
