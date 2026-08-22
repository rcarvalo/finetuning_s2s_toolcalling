"""``AudioResampler`` — rééchantillonnage, isolé derrière sa propre frontière.

torchaudio est importé **en tête** : ce module est donc lourd, et n'est chargé
que par qui rééchantillonne réellement. C'est ce qui permet à
:class:`~lfm2_audio.ds.audio.Waveform` de rester du numpy pur — et donc à
``lfm2-evaluate --list-scorers`` de tourner sur une machine sans torch.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio


class AudioResampler:
    """Convertit un signal d'une fréquence à une autre."""

    def resample(self, samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return samples
        converted = torchaudio.functional.resample(torch.from_numpy(samples), source_rate, target_rate)
        return np.asarray(converted.numpy(), dtype=np.float32)
