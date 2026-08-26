"""Tests for the Inspect bridge: dataset, provider glue, scorer adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.evaluation.question import Question

pytest.importorskip("inspect_ai")

from inspect_ai.model import ChatMessageUser, ContentAudio
from inspect_ai.model import ContentText as InspectText

from lfm2_audio.inspect_bridge.audio import data_uri_to_waveform, waveform_to_data_uri
from lfm2_audio.inspect_bridge.dataset import question_set_dataset, resolve_dataset_path, to_sample
from lfm2_audio.inspect_bridge.provider import _last_user_turn
from lfm2_audio.inspect_bridge.scores import to_inspect_score
from lfm2_audio.scorer.result import ScoreResult

# --------------------------------------------------------------------------- #
# Audio : le viewer n'accepte qu'une data URI
# --------------------------------------------------------------------------- #


def test_should_round_trip_a_waveform_through_a_data_uri() -> None:
    original = Waveform.of(np.linspace(-0.5, 0.5, 2400, dtype=np.float32), 24_000)

    restored = data_uri_to_waveform(waveform_to_data_uri(original))

    assert restored.sample_rate == 24_000
    assert restored.samples.size == 2400
    assert np.allclose(restored.samples, original.samples, atol=1e-4)


def test_data_uri_should_declare_the_mime_type_the_viewer_checks() -> None:
    """`isRenderableAudioSource` refuses anything else, and shows an inert link."""
    uri = waveform_to_data_uri(Waveform.of(np.zeros(240, dtype=np.float32), 24_000))

    assert uri.startswith("data:audio/wav;base64,")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def test_should_send_a_spoken_question_as_audio(tmp_path: Path) -> None:
    """The model must hear the question; a transcript would let it read instead."""
    wav = tmp_path / "q.wav"
    Waveform.of(np.zeros(1600, dtype=np.float32), 16_000).save(wav)
    question = Question(question_id="c1", text="transcript", audio_path=wav)

    sample = to_sample(question)

    assert isinstance(sample.input, list)
    parts = sample.input[0].content
    assert [type(p) for p in parts] == [ContentAudio]
    assert sample.metadata is not None
    assert sample.metadata["prompt_text"] == "transcript"


def test_should_send_a_written_question_as_text() -> None:
    sample = to_sample(Question(question_id="c1", text="What is the weather?"))

    assert sample.input == "What is the weather?"


def test_should_carry_the_expected_calls_into_metadata() -> None:
    expected = [{"name": "web_search", "arguments": {"query": "x"}}]

    sample = to_sample(Question(question_id="c1", text="q", expected_calls=expected))

    assert sample.metadata is not None
    assert sample.metadata["expected_calls"] == expected


def test_should_load_a_question_set_relative_to_the_repo(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"id": "c1", "turns": [{"role": "user", "text": "hello"}]}))

    dataset = question_set_dataset(str(path), limit=1)

    assert len(dataset) == 1


def test_should_say_where_it_looked_when_the_dataset_is_missing() -> None:
    with pytest.raises(FileNotFoundError, match="introuvable"):
        resolve_dataset_path("nowhere/at/all.jsonl")


# --------------------------------------------------------------------------- #
# Provider : quelle question part au modèle
# --------------------------------------------------------------------------- #


def test_should_prefer_the_audio_over_its_transcript() -> None:
    audio = waveform_to_data_uri(Waveform.of(np.zeros(1600, dtype=np.float32), 16_000))
    message = ChatMessageUser(content=[InspectText(text="transcript"), ContentAudio(audio=audio, format="wav")])

    text, waveform = _last_user_turn([message])

    assert text is None
    assert waveform is not None


def test_should_pass_a_written_question_as_text() -> None:
    text, waveform = _last_user_turn([ChatMessageUser(content="hello")])

    assert (text, waveform) == ("hello", None)


# --------------------------------------------------------------------------- #
# Scores : ne jamais confondre « pas mesuré » et « mal mesuré »
# --------------------------------------------------------------------------- #


def test_should_translate_a_measured_result() -> None:
    score = to_inspect_score(ScoreResult.ok("utmos", 4.12, details={"duration_s": 3.0}))

    assert score is not None
    assert score.value == 4.12
    assert score.metadata == {"duration_s": 3.0}


@pytest.mark.parametrize(
    "result",
    [
        ScoreResult.unavailable("nisqa", "poids absents"),
        ScoreResult.skipped("wer", "aucun audio"),
        ScoreResult.failed("dnsmos", "boom"),
    ],
)
def test_should_refuse_to_turn_an_unmeasured_result_into_a_score(result: ScoreResult) -> None:
    assert to_inspect_score(result) is None
