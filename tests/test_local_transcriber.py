"""Tests for ``bench.local_transcriber.LocalTranscriber``.

faster-whisper runs on CPU and pulls no torch, which is what keeps the voice
client (and the WER scorer) off a GPU that is already saturated by the audio
model itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from lfm2_audio.bench.local_transcriber import LocalTranscriber
from lfm2_audio.ds.audio import INPUT_SAMPLE_RATE, Waveform


class _Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisper:
    def __init__(self, *segments: str) -> None:
        self._segments = segments
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, samples: Any, **kwargs: Any) -> tuple[list[_Segment], None]:
        self.calls.append({"samples": samples, **kwargs})
        return [_Segment(s) for s in self._segments], None


def _wave(rate: int = INPUT_SAMPLE_RATE, seconds: float = 0.5) -> Waveform:
    return Waveform(np.zeros(int(rate * seconds), dtype=np.float32), rate)


def test_should_join_segments_into_one_line() -> None:
    transcriber = LocalTranscriber()
    transcriber._model = _FakeWhisper(" hello", " world ")

    assert transcriber.transcribe(_wave()) == "hello  world"


def test_should_resample_to_the_encoder_rate() -> None:
    # L'encodeur est calibré 16 kHz : transcrire à un autre taux dégrade le
    # WER sans jamais lever d'erreur.
    fake = _FakeWhisper("ok")
    transcriber = LocalTranscriber()
    transcriber._model = fake

    transcriber.transcribe(_wave(rate=24_000, seconds=1.0))

    assert len(fake.calls[0]["samples"]) == pytest.approx(INPUT_SAMPLE_RATE, rel=0.05)


def test_should_reuse_the_loaded_model() -> None:
    fake = _FakeWhisper("ok")
    transcriber = LocalTranscriber()
    transcriber._model = fake

    transcriber.transcribe(_wave())
    transcriber.transcribe(_wave())

    assert transcriber._model is fake


def test_should_return_empty_when_faster_whisper_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Une dépendance optionnelle manquante ne doit pas casser un tour de voix :
    # pas de transcription vaut mieux qu'une exception au milieu d'une réponse.
    monkeypatch.setattr(LocalTranscriber, "_load", lambda self: None)

    assert LocalTranscriber().transcribe(_wave()) == ""
