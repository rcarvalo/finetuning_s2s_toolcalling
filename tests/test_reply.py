"""Tests du value object ``Reply`` et de ses métriques."""

from __future__ import annotations

import numpy as np
import pytest

from lfm2_audio.ds.audio import OUTPUT_SAMPLE_RATE, Waveform
from lfm2_audio.ds.reply import Reply, TurnMetrics


def _audio(seconds: float = 2.0) -> Waveform:
    return Waveform.of(np.zeros(int(OUTPUT_SAMPLE_RATE * seconds), dtype=np.float32), OUTPUT_SAMPLE_RATE)


def test_should_unpack_like_a_tuple():
    audio = _audio()

    text, unpacked = Reply(text="hello", audio=audio)

    assert text == "hello"
    assert unpacked is audio


def test_should_report_absence_of_audio():
    assert not Reply(text="hi").has_audio
    assert not Reply(text="hi", audio=Waveform.of(np.array([]), OUTPUT_SAMPLE_RATE)).has_audio
    assert Reply(text="hi", audio=_audio()).has_audio


def test_real_time_factor_should_compare_generation_to_audio_duration():
    reply = Reply(text="hi", audio=_audio(2.0), metrics=TurnMetrics(total_s=1.0))

    assert reply.real_time_factor == pytest.approx(0.5)


def test_real_time_factor_should_be_none_without_audio():
    assert Reply(text="hi", metrics=TurnMetrics(total_s=1.0)).real_time_factor is None


def test_save_audio_should_write_the_file(tmp_path):
    reply = Reply(text="hi", audio=_audio(0.1))

    path = reply.save_audio(tmp_path / "reply.wav")

    assert path is not None
    assert path.exists()


def test_save_audio_should_return_none_without_audio(tmp_path):
    assert Reply(text="hi").save_audio(tmp_path / "reply.wav") is None
    assert not (tmp_path / "reply.wav").exists()


def test_raw_text_should_keep_markers_while_text_is_clean():
    """L'orchestrateur parse ``raw_text`` ; l'UI affiche ``text``."""
    reply = Reply(text="Checking.", raw_text="Checking.<|tool_call_start|>[f()]<|tool_call_end|>")

    assert "<|tool_call_start|>" in reply.raw_text
    assert "<|" not in reply.text


def test_metrics_should_serialise_for_logging():
    metrics = TurnMetrics(ttfa_s=0.31, total_s=1.2, audio_frames=42)

    assert metrics.as_dict() == {"ttfa_s": 0.31, "total_s": 1.2, "audio_frames": 42}


def test_metrics_should_default_to_no_measurement():
    metrics = TurnMetrics()

    assert metrics.ttfa_s is None
    assert metrics.audio_frames == 0
