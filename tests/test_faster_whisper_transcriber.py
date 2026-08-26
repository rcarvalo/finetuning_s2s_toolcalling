"""``FasterWhisperTranscriber`` — l'ASR du WER, sur CPU.

Le chemin ``transformers`` charge Whisper sur ``cuda:0``. Le GPU est déjà
pris : l'entraînement culmine à ~96 % de la VRAM d'une L4 et une campagne y
tient le modèle audio. CTranslate2 en int8 rend donc le WER mesurable là où
il ne l'était pas — pendant un entraînement, et sur un portable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from lfm2_audio.ds.audio import INPUT_SAMPLE_RATE, Waveform
from lfm2_audio.scorer.audio.faster_whisper_transcriber import FasterWhisperTranscriber
from lfm2_audio.scorer.audio.transcriber import Transcriber


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


def test_should_satisfy_the_transcriber_protocol() -> None:
    assert isinstance(FasterWhisperTranscriber(), Transcriber)


def test_should_default_to_cpu_int8() -> None:
    # Le défaut est ce qui compte : un défaut CUDA ferait déborder le GPU sans
    # que la config ait rien demandé.
    transcriber = FasterWhisperTranscriber()

    assert transcriber._device == "cpu"
    assert transcriber._compute_type == "int8"


def test_should_join_segments_into_one_line() -> None:
    transcriber = FasterWhisperTranscriber()
    transcriber._model = _FakeWhisper(" hello", " world ")

    assert transcriber.transcribe(_wave()) == "hello  world"


def test_should_resample_to_the_encoder_rate() -> None:
    fake = _FakeWhisper("ok")
    transcriber = FasterWhisperTranscriber()
    transcriber._model = fake

    transcriber.transcribe(_wave(rate=24_000, seconds=1.0))

    assert len(fake.calls[0]["samples"]) == pytest.approx(INPUT_SAMPLE_RATE, rel=0.05)


def test_should_report_the_model_it_used() -> None:
    # Deux campagnes transcrites par des ASR différents ne sont pas comparables.
    assert FasterWhisperTranscriber("small").model_id == "faster-whisper/small"


def test_should_return_empty_when_faster_whisper_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FasterWhisperTranscriber, "_load", lambda self: None)

    assert FasterWhisperTranscriber().transcribe(_wave()) == ""
