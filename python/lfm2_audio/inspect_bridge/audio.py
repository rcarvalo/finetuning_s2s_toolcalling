"""Waveform → the one audio form the Inspect viewer will actually play.

Its front-end validates the source with ``isRenderableAudioSource``, which
requires a base64 data URI whose MIME type matches the declared format; a file
path renders as an inert reference instead of a player. So the WAV travels
inside the log, as Inspect already does for images.

Cost is real — base64 inflates by a third — so a caller grading hundreds of
long replies should expect a log measured in hundreds of megabytes.
"""

from __future__ import annotations

import base64
import io
import wave

import numpy as np

from lfm2_audio.ds.audio import Waveform

WAV_MIME = "audio/wav"
_PCM16_MAX = 32767


def waveform_to_data_uri(waveform: Waveform) -> str:
    """Encode a waveform as a ``data:audio/wav;base64,…`` URI."""
    buffer = io.BytesIO()
    pcm16 = (np.clip(waveform.samples, -1.0, 1.0) * _PCM16_MAX).astype(np.int16)
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(waveform.sample_rate)
        handle.writeframes(pcm16.tobytes())
    return f"data:{WAV_MIME};base64," + base64.b64encode(buffer.getvalue()).decode()


def wav_file_to_data_uri(path: str) -> str:
    """Same, straight from a WAV file — no decode/re-encode round trip."""
    with open(path, "rb") as handle:
        return f"data:{WAV_MIME};base64," + base64.b64encode(handle.read()).decode()
