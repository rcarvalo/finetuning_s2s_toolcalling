"""Tests of the generated-speech extraction from Inspect logs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from inspect_ai.log import EvalLog, EvalSpec, write_eval_log
from inspect_ai.log._log import EvalSample as LogSample
from inspect_ai.model import ChatMessageUser, ContentAudio, ModelOutput

from lfm2_audio.ds.audio import OUTPUT_SAMPLE_RATE, Waveform
from lfm2_audio.evaluation.eval_log_audio import LoggedReply, extract_replies, latest_log
from lfm2_audio.inspect_bridge.audio import waveform_to_data_uri


def _log_with_audio(tmp_path: Path, *, with_audio: bool = True) -> Path:
    waveform = Waveform.of(np.zeros(OUTPUT_SAMPLE_RATE // 2, dtype=np.float32), OUTPUT_SAMPLE_RATE)
    output = ModelOutput.from_content("mockllm/model", "Bonjour tout le monde<|text_end|>")
    if with_audio:
        output.choices[0].message.content = [
            ContentAudio(audio=waveform_to_data_uri(waveform), format="wav"),
        ]
    sample = LogSample(
        id="s1",
        epoch=0,
        input=[ChatMessageUser(content="question")],
        target="",
        messages=[],
        output=output,
        metadata={"lang": "fr"},
    )
    log = EvalLog(
        eval=EvalSpec(
            created="2026-08-27T00:00:00",
            task="voice_eval",
            model="mockllm/model",
            dataset={},
            config={},
        ),
        samples=[sample],
    )
    path = tmp_path / "run.eval"
    write_eval_log(log, str(path))
    return path


def test_latest_log_should_pick_the_most_recent_attempt(tmp_path: Path) -> None:
    """Campaigns are re-run after fixes and attempts share a directory."""
    old = tmp_path / "old.eval"
    old.write_text("", encoding="utf-8")
    import os
    import time

    time.sleep(0.01)
    new = tmp_path / "new.eval"
    new.write_text("", encoding="utf-8")
    os.utime(old, (1, 1))

    assert latest_log(tmp_path) == new


def test_latest_log_should_raise_when_the_directory_holds_none(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"aucun \.eval"):
        latest_log(tmp_path)


def test_should_write_a_wav_per_reply_and_report_its_duration(tmp_path: Path) -> None:
    log_path = _log_with_audio(tmp_path)

    replies = list(extract_replies(log_path, tmp_path / "audio"))

    assert len(replies) == 1
    reply = replies[0]
    assert reply.wav_path.exists()
    assert sf.info(str(reply.wav_path)).samplerate == OUTPUT_SAMPLE_RATE
    assert reply.seconds == pytest.approx(0.5, abs=0.01)
    assert reply.question_language == "fr"


def test_should_skip_a_reply_that_carries_no_audio(tmp_path: Path) -> None:
    """A turn can be text-only; the gate pass must not invent a file for it."""
    log_path = _log_with_audio(tmp_path, with_audio=False)

    assert list(extract_replies(log_path, tmp_path / "audio")) == []


def test_spoken_text_should_drop_the_markers(tmp_path: Path) -> None:
    reply = LoggedReply(
        sample_id="s1",
        text="Bonjour tout le monde<|text_end|>",
        wav_path=tmp_path / "s1.wav",
        seconds=1.0,
        question_language="fr",
    )

    assert reply.spoken_text == "Bonjour tout le monde"
