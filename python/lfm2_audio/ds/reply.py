"""Résultat d'un tour de génération."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lfm2_audio.ds.audio import Waveform


@dataclass(frozen=True, slots=True)
class TurnMetrics:
    """Latences d'un tour. ``None`` = l'étape n'a pas eu lieu.

    ``audio_frames`` est la vérité terrain du stage 0 : des frames émises sans
    audio en sortie désignent la plomberie stage 0 → stage 1, alors que zéro
    frame désigne le prompt ou le modèle. Cette distinction évite de déboguer au
    mauvais endroit.
    """

    ttfa_s: float | None = None
    """Time-to-first-audio : premier chunk réellement décodé."""

    total_s: float = 0.0
    audio_frames: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"ttfa_s": self.ttfa_s, "total_s": self.total_s, "audio_frames": self.audio_frames}


@dataclass(frozen=True, slots=True)
class Reply:
    """Réponse du modèle : texte, audio et métriques.

    Déballable comme un tuple pour l'usage courant :

    >>> text, audio = model.reply(text="Hello")  # doctest: +SKIP
    """

    text: str
    audio: Waveform | None = None
    metrics: TurnMetrics = TurnMetrics()
    raw_text: str = ""
    """Texte brut du stage 0, marqueurs ``<|…|>`` compris (parsing des tool calls)."""

    def __iter__(self) -> Iterator[Any]:
        return iter((self.text, self.audio))

    @property
    def has_audio(self) -> bool:
        return self.audio is not None and not self.audio.is_empty

    @property
    def real_time_factor(self) -> float | None:
        """``total / durée d'audio`` — > 1 : la lecture rattrape la génération."""
        if self.audio is None or self.audio.is_empty:
            return None
        return self.metrics.total_s / self.audio.duration_s

    def save_audio(self, path: str | Path) -> Path | None:
        """Écrit l'audio s'il existe. Retourne le chemin, ou ``None``."""
        if self.audio is None or self.audio.is_empty:
            return None
        return self.audio.save(path)
