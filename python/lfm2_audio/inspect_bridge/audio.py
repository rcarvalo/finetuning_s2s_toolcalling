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


def data_uri_to_waveform(source: str) -> Waveform:
    """Decode what a log carries back into a waveform.

    Accepts both forms a ``ContentAudio`` can hold: the base64 data URI the
    viewer needs, and a plain path — a dataset written by hand references files.
    """
    if not source.startswith("data:"):
        return Waveform.from_file(source)
    payload = base64.b64decode(source.split(",", 1)[1])
    # Decoded with the stdlib rather than handed to `Waveform.from_file`, which
    # stringifies its argument and would turn a buffer into a bogus path. WAV by
    # construction here, symmetric with the encoder above.
    with wave.open(io.BytesIO(payload), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        sample_rate = handle.getframerate()
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / (_PCM16_MAX + 1)
    return Waveform.of(samples, sample_rate)
