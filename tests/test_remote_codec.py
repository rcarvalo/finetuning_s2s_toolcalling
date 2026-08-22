"""Codec WAV base64 du contrat client ↔ handler serverless."""

from __future__ import annotations

import base64
import io
import wave

import numpy as np
import pytest

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.remote.codec import AudioCodecError, waveform_from_wav_b64, waveform_to_wav_b64


def _sine(rate: int = 16_000, seconds: float = 0.1) -> Waveform:
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    return Waveform.of(0.5 * np.sin(2 * np.pi * 440 * t), rate)


def test_should_roundtrip_waveform_through_b64() -> None:
    original = _sine()

    decoded = waveform_from_wav_b64(waveform_to_wav_b64(original))

    assert decoded.sample_rate == original.sample_rate
    assert decoded.samples.shape == original.samples.shape
    np.testing.assert_allclose(decoded.samples, original.samples, atol=1.5 / 32_767)


def test_should_clip_out_of_range_samples_when_encoding() -> None:
    loud = Waveform.of(np.array([2.0, -2.0], dtype=np.float32), 16_000)

    decoded = waveform_from_wav_b64(waveform_to_wav_b64(loud))

    assert np.all(np.abs(decoded.samples) <= 1.0)


def test_should_downmix_stereo_wav_to_mono_when_decoding() -> None:
    frames = np.array([[1000, 3000], [-2000, -4000]], dtype=np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(frames.tobytes())

    decoded = waveform_from_wav_b64(base64.b64encode(buffer.getvalue()).decode("ascii"))

    assert decoded.samples.shape == (2,)
    np.testing.assert_allclose(decoded.samples[0], 2000 / 32_768, atol=1e-4)


def test_should_raise_when_payload_is_not_base64() -> None:
    with pytest.raises(AudioCodecError, match="base64"):
        waveform_from_wav_b64("cec¡ n'est pas du base64")


def test_should_raise_when_payload_is_not_a_wav() -> None:
    payload = base64.b64encode(b"pas un fichier wav du tout").decode("ascii")

    with pytest.raises(AudioCodecError, match="illisible"):
        waveform_from_wav_b64(payload)


def test_should_raise_when_wav_is_not_pcm16() -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)  # PCM 8 bits
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x7f\xff")

    with pytest.raises(AudioCodecError, match="PCM16"):
        waveform_from_wav_b64(base64.b64encode(buffer.getvalue()).decode("ascii"))
