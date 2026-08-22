"""Sérialisation WAV ↔ base64 du contrat handler serverless / client.

Volontairement en stdlib (``wave``) + numpy : le client tourne sur le Reachy
Mini sans soundfile ni torch. PCM 16 bits mono — le format que produit déjà
:meth:`Waveform.save` et qu'attend l'encodeur du modèle.
"""

from __future__ import annotations

import base64
import binascii
import io
import wave

import numpy as np

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.ds.audio import Waveform

_PCM16_MAX = 32_767


class AudioCodecError(Lfm2AudioError):
    """Payload audio illisible (base64 invalide ou WAV non PCM16 mono)."""


def waveform_to_wav_b64(waveform: Waveform) -> str:
    """Encode un :class:`Waveform` en WAV PCM16 mono, retourné en base64."""
    pcm16 = (np.clip(waveform.samples, -1.0, 1.0) * _PCM16_MAX).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(waveform.sample_rate)
        handle.writeframes(pcm16.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def waveform_from_wav_b64(data: str) -> Waveform:
    """Décode un WAV base64 (PCM entier, mono ou multi-canaux moyennés)."""
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        message = "payload audio_b64 : base64 invalide"
        raise AudioCodecError(message) from exc
    try:
        with wave.open(io.BytesIO(raw), "rb") as handle:
            if handle.getsampwidth() != 2:
                message = f"WAV non PCM16 (sampwidth={handle.getsampwidth()})"
                raise AudioCodecError(message)
            frames = handle.readframes(handle.getnframes())
            channels = handle.getnchannels()
            rate = handle.getframerate()
    except wave.Error as exc:
        message = "payload audio_b64 : WAV illisible"
        raise AudioCodecError(message) from exc
    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return Waveform.from_pcm16(pcm, rate)
