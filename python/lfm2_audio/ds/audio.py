"""``Waveform`` — value object d'un signal audio.

Remplace le ``tuple[np.ndarray, int]`` qui circulait dans tout le projet : la
fréquence d'échantillonnage voyage désormais AVEC le signal, ce qui rend
impossible la classe de bug la plus coûteuse rencontrée ici — envoyer du 48 kHz
à un encodeur mel calibré 16 kHz, qui « entend » alors du charabia et répond à
côté sans jamais lever d'erreur.

Immuable : chaque transformation retourne un nouveau ``Waveform``.
"""

from __future__ import annotations

import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

INPUT_SAMPLE_RATE = 16_000
"""Fréquence attendue par l'encodeur conformer (``preprocessor.sample_rate``)."""

OUTPUT_SAMPLE_RATE = 24_000
"""Fréquence produite par le détokeniseur Mimi."""

_PCM16_MAX = 32_767


@dataclass(frozen=True, slots=True)
class Waveform:
    """Signal mono float32 et sa fréquence d'échantillonnage.

    Construire via :meth:`of` plutôt que directement : le constructeur ne
    normalise rien, ``of`` garantit l'invariant « mono, float32, 1-D ».
    """

    samples: np.ndarray
    sample_rate: int

    # -- construction ---------------------------------------------------- #

    @classmethod
    def of(cls, samples: npt.ArrayLike, sample_rate: int) -> Self:
        """Normalise n'importe quel signal en mono float32 1-D."""
        if sample_rate <= 0:
            message = f"sample_rate doit être > 0, reçu {sample_rate}"
            raise ValueError(message)
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim > 1:
            # (T, C) ou (C, T) : l'axe des canaux est le plus court.
            array = array.mean(axis=int(np.argmin(array.shape)))
        return cls(samples=array.reshape(-1), sample_rate=sample_rate)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Lit un fichier audio (WAV, FLAC, …) via soundfile."""
        import soundfile as sf

        data, sample_rate = sf.read(str(path), dtype="float32")
        return cls.of(data, int(sample_rate))

    @classmethod
    def from_pcm16(cls, pcm: np.ndarray, sample_rate: int) -> Self:
        """Convertit un buffer entier 16 bits (micro navigateur, WebRTC)."""
        array = np.asarray(pcm)
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / (_PCM16_MAX + 1)
        return cls.of(array, sample_rate)

    @classmethod
    def concat(cls, chunks: Sequence[Self]) -> Self | None:
        """Concatène des chunks issus du streaming. ``None`` si aucun n'a de contenu.

        La fréquence vient des chunks eux-mêmes ; les mélanger produirait un son
        accéléré ou ralenti sans erreur visible, donc on refuse explicitement.
        """
        usable = [chunk for chunk in chunks if not chunk.is_empty]
        if not usable:
            return None
        rates = {chunk.sample_rate for chunk in usable}
        if len(rates) > 1:
            message = f"chunks de fréquences hétérogènes : {sorted(rates)}"
            raise ValueError(message)
        return cls(
            samples=np.concatenate([chunk.samples for chunk in usable]),
            sample_rate=usable[0].sample_rate,
        )

    # -- propriétés ------------------------------------------------------ #

    @property
    def duration_s(self) -> float:
        return float(self.samples.size) / self.sample_rate

    @property
    def rms(self) -> float:
        """Énergie RMS — sert à écarter l'écho et le silence en entrée micro."""
        return float(np.sqrt(np.mean(self.samples**2))) if self.samples.size else 0.0

    @property
    def is_empty(self) -> bool:
        return self.samples.size == 0

    # -- transformations ------------------------------------------------- #

    def resample(self, target_rate: int) -> Self:
        """Rééchantillonne (no-op si déjà à la bonne fréquence)."""
        if self.sample_rate == target_rate:
            return self

        import torch
        import torchaudio

        resampled = torchaudio.functional.resample(torch.from_numpy(self.samples), self.sample_rate, target_rate)
        return type(self)(samples=resampled.numpy().astype(np.float32), sample_rate=target_rate)

    def for_encoder(self) -> Self:
        """Prêt pour l'encodeur audio : mono 16 kHz.

        À appeler EN AMONT de vLLM — le data-parser multimodal ne rééchantillonne
        pas, et un mel calculé sur du 48 kHz est silencieusement faux.
        """
        return self.resample(INPUT_SAMPLE_RATE)

    # -- entrées / sorties ------------------------------------------------ #

    def save(self, path: str | Path) -> Path:
        """Écrit un WAV PCM 16 bits mono. Crée le répertoire parent au besoin."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pcm16 = (np.clip(self.samples, -1.0, 1.0) * _PCM16_MAX).astype(np.int16)
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(pcm16.tobytes())
        return destination

    def as_model_input(self) -> tuple[np.ndarray, int]:
        """Tuple attendu par ``multi_modal_data`` de vLLM et par liquid-audio."""
        return self.samples, self.sample_rate
