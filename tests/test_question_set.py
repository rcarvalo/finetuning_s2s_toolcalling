"""Tests du chargement du jeu de questions."""

from __future__ import annotations

import json

import pytest

from lfm2_audio.ds.dialogue import DialogueValidationError
from lfm2_audio.evaluation.question_set import QuestionSet

BENCHMARK = "benchmark/toolcalling_en/cases.sample.jsonl"


def _write(tmp_path, dialogues):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in dialogues) + "\n", encoding="utf-8")
    return path


def test_should_load_the_shipped_english_benchmark():
    questions = QuestionSet.from_jsonl(BENCHMARK)

    assert len(questions) > 0
    assert questions.positives > 0
    assert questions.source == BENCHMARK


def test_should_extract_question_and_expected_call(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "id": "q1",
                "turns": [
                    {"role": "user", "text": "weather in tokyo?"},
                    {
                        "role": "assistant",
                        "tool_calls": [{"name": "web_search", "arguments": {"query": "tokyo"}}],
                    },
                ],
            }
        ],
    )

    question = QuestionSet.from_jsonl(path).questions[0]

    assert question.question_id == "q1"
    assert question.text == "weather in tokyo?"
    assert question.expects_tool_call
    assert question.expected_calls[0]["name"] == "web_search"


def test_negative_case_should_expect_no_call(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "id": "n1",
                "turns": [
                    {"role": "user", "text": "hello"},
                    {"role": "assistant", "text": "Hi there!"},
                ],
            }
        ],
    )

    question = QuestionSet.from_jsonl(path).questions[0]

    assert not question.expects_tool_call
    assert question.reference_answer == "Hi there!"


def test_audio_path_should_be_resolved_against_the_root(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "id": "a1",
                "turns": [
                    {"role": "user", "audio": "clip.wav"},
                    {"role": "assistant", "text": "ok"},
                ],
            }
        ],
    )

    question = QuestionSet.from_jsonl(path, audio_root=tmp_path / "wavs").questions[0]

    assert question.audio_path == tmp_path / "wavs" / "clip.wav"


def test_a_malformed_line_should_be_rejected(tmp_path):
    """Le jeu passe par la validation pydantic des dialogues : une éval ne doit
    pas tourner sur un fichier à moitié corrompu."""
    path = _write(tmp_path, [{"id": "bad", "turns": [{"role": "wizard", "text": "x"}]}])

    with pytest.raises(DialogueValidationError):
        QuestionSet.from_jsonl(path)


def test_take_should_truncate():
    full = QuestionSet.from_jsonl(BENCHMARK)

    assert len(full.take(3)) == 3
    assert len(full.take(None)) == len(full)
    assert len(full.take(10_000)) == len(full)


def test_filter_ids_should_keep_only_the_requested_cases():
    full = QuestionSet.from_jsonl(BENCHMARK)
    wanted = full.questions[0].question_id

    filtered = full.filter_ids([wanted])

    assert len(filtered) == 1
    assert filtered.questions[0].question_id == wanted
