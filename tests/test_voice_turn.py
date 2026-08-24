"""Tests for ``bench.voice_turn.VoiceTurnHandler``."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from lfm2_audio.bench.voice_turn import VoiceTurnHandler
from lfm2_audio.ds.audio import Waveform


class _FakeClient:
    """Records invoke_stream calls and streams two canned 24 kHz chunks."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.last_reply = None

    def invoke_stream(self, *, audio: Waveform, max_tokens: int | None = None) -> Iterator[Waveform]:
        self.calls.append({"audio": audio, "max_tokens": max_tokens})
        yield Waveform.of(np.full(240, 0.1, dtype=np.float32), 24_000)
        yield Waveform.of(np.full(480, 0.2, dtype=np.float32), 24_000)


@pytest.fixture
def fake_client() -> _FakeClient:
    return _FakeClient()


def _speech(sample_rate: int = 48_000, duration_s: float = 0.5) -> tuple[int, np.ndarray]:
    length = int(sample_rate * duration_s)
    tone = (np.sin(np.linspace(0.0, 200.0, length)) * 12_000).astype(np.int16)
    return sample_rate, tone


def test_should_stream_chunks_as_they_arrive(fake_client: _FakeClient) -> None:
    handler = VoiceTurnHandler(fake_client)  # type: ignore[arg-type]

    chunks = list(handler.respond(_speech()))

    assert [rate for rate, _ in chunks] == [24_000, 24_000]
    assert [samples.size for _, samples in chunks] == [240, 480]


def test_should_send_native_rate_mono_to_the_endpoint(fake_client: _FakeClient) -> None:
    """No client-side resampling: the worker resamples on the GPU."""
    handler = VoiceTurnHandler(fake_client)  # type: ignore[arg-type]

    list(handler.respond(_speech(sample_rate=48_000)))

    sent = fake_client.calls[0]["audio"]
    assert isinstance(sent, Waveform)
    assert sent.sample_rate == 48_000
    assert sent.samples.ndim == 1


def test_should_forward_max_tokens(fake_client: _FakeClient) -> None:
    handler = VoiceTurnHandler(fake_client, max_tokens=256)  # type: ignore[arg-type]

    list(handler.respond(_speech()))

    assert fake_client.calls[0]["max_tokens"] == 256


def test_should_discard_silence_without_calling_the_endpoint(fake_client: _FakeClient) -> None:
    handler = VoiceTurnHandler(fake_client)  # type: ignore[arg-type]
    silence = (48_000, np.zeros(24_000, dtype=np.int16))

    chunks = list(handler.respond(silence))

    assert chunks == []
    assert fake_client.calls == []


def test_should_save_user_and_reply_wavs_when_save_dir_set(fake_client: _FakeClient, tmp_path: Path) -> None:
    handler = VoiceTurnHandler(fake_client, save_dir=tmp_path)  # type: ignore[arg-type]

    list(handler.respond(_speech()))

    saved = sorted(p.name for p in tmp_path.glob("*.wav"))
    assert len(saved) == 1  # fake client has no last_reply → user WAV only
    assert saved[0].endswith("_user.wav")
