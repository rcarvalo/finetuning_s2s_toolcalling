"""Pieces both demo UIs share to move a turn's audio.

Kept apart so the simple UI never pulls the WebRTC stack: ``fastrtc`` is only
needed by the hands-free path, and a browser tunnel demo must stay runnable on
a machine that has none of it.
"""

from __future__ import annotations

import os
import threading

import numpy as np

SR_OUT = 24_000
# Sous ce RMS : écho (sortie modèle reprise au micro) ou silence → on ignore.
MIN_INPUT_RMS = float(os.environ.get("MIN_INPUT_RMS", "0.03"))
# Un seul modèle sur un seul GPU : les tours ne se chevauchent jamais.
LOCK = threading.Lock()


def as_pcm16(samples: np.ndarray) -> tuple[int, np.ndarray]:
    """Float32 mono to the (rate, int16) pair both gradio and fastrtc expect."""
    return SR_OUT, (np.clip(samples, -1.0, 1.0) * 32_767).astype(np.int16)
