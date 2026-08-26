"""Tests for ``evaluation.sample_archive.SampleArchive``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.evaluation.sample_archive import SampleArchive
from lfm2_audio.scorer.sample import EvalSample


def _sample(sample_id: str = "case_1", *, with_audio: bool = True) -> EvalSample:
    tone = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
    return EvalSample(
        sample_id=sample_id,
        prompt_text="what is the weather",
        predicted_text="It is sunny.",
        predicted_audio=Waveform.of(tone, 24_000) if with_audio else None,
        reference_text="It is sunny in Paris.",
        expected_calls=[{"name": "web_search", "arguments": {"query": "weather"}}],
        tool_results=[{"name": "web_search", "content": "sunny"}],
        metadata={"source": "unit-test"},
    )


def test_should_round_trip_a_sample_with_audio(tmp_path: Path) -> None:
    archive = SampleArchive(tmp_path)
    archive.save(_sample())

    restored = list(archive.load())

    assert len(restored) == 1
    assert restored[0].sample_id == "case_1"
    assert restored[0].predicted_text == "It is sunny."
    assert restored[0].reference_text == "It is sunny in Paris."
    assert restored[0].expected_calls[0]["name"] == "web_search"
    assert restored[0].tool_results[0]["content"] == "sunny"
    assert restored[0].metadata == {"source": "unit-test"}
    assert restored[0].has_predicted_audio


def test_should_preserve_the_audio_sample_rate(tmp_path: Path) -> None:
    """A 24 kHz reply reloaded as 16 kHz would silently change every MOS."""
    archive = SampleArchive(tmp_path)
    archive.save(_sample())

    audio = next(iter(archive.load())).predicted_audio

    assert audio is not None
    assert audio.sample_rate == 24_000
    assert audio.samples.size == 2400


def test_should_round_trip_a_sample_without_audio(tmp_path: Path) -> None:
    archive = SampleArchive(tmp_path)
    archive.save(_sample(with_audio=False))

    restored = next(iter(archive.load()))

    assert restored.predicted_audio is None
    assert not restored.has_predicted_audio


def test_should_load_samples_in_a_stable_order(tmp_path: Path) -> None:
    archive = SampleArchive(tmp_path)
    for name in ("case_3", "case_1", "case_2"):
        archive.save(_sample(name))

    assert [s.sample_id for s in archive.load()] == ["case_1", "case_2", "case_3"]


def test_should_report_an_empty_archive(tmp_path: Path) -> None:
    assert len(SampleArchive(tmp_path / "nothing")) == 0
