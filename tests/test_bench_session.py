"""Tests for BenchSession against a fake answer source (no GPU, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from lfm2_audio.bench.session import BenchSession
from lfm2_audio.bench.source import AnswerSource
from lfm2_audio.bench.store import RatingStore
from lfm2_audio.ds.audio import OUTPUT_SAMPLE_RATE, Waveform
from lfm2_audio.ds.reply import Reply
from lfm2_audio.evaluation.question import Question
from lfm2_audio.evaluation.question_set import QuestionSet


class FakeSource:
    """Records what it was asked. Satisfies the ``AnswerSource`` protocol."""

    def __init__(self, *, label: str = "fake@test", stateful: bool = True) -> None:
        self._label = label
        self._stateful = stateful
        self.resets = 0
        self.calls: list[str | None] = []

    @property
    def label(self) -> str:
        return self._label

    @property
    def keeps_history(self) -> bool:
        return self._stateful

    def answer(self, *, text=None, audio=None, max_tokens=400) -> Reply:
        self.calls.append(text)
        return Reply(
            text="a spoken answer",
            audio=Waveform.of(np.ones(OUTPUT_SAMPLE_RATE, dtype=np.float32) * 0.1, OUTPUT_SAMPLE_RATE),
        )

    def reset(self) -> None:
        self.resets += 1


def _questions() -> QuestionSet:
    return QuestionSet(
        questions=(
            Question(question_id="q1", text="first question"),
            Question(question_id="q2", text="second question"),
        ),
        source="unit-test",
    )


@pytest.fixture
def session(tmp_path) -> BenchSession:
    return BenchSession(
        FakeSource(),
        _questions(),
        store=RatingStore(tmp_path / "r.jsonl"),
        audio_dir=tmp_path / "audio",
    )


def test_a_fake_source_satisfies_the_protocol():
    assert isinstance(FakeSource(), AnswerSource)


def test_version_should_default_to_the_source_label(session):
    assert session.version == "fake@test"


def test_explicit_version_should_win(tmp_path):
    bench = BenchSession(FakeSource(), _questions(), version="run-42", store=RatingStore(tmp_path / "r.jsonl"))

    assert bench.version == "run-42"


def test_generate_should_reset_between_cases(session):
    """Each case is judged on its own; leftover context would make the answer
    depend on which cases happened to be rated before it."""
    session.generate("q1")
    session.generate("q2")

    assert session._source.resets == 2


def test_generate_should_persist_the_audio(session):
    _, path = session.generate("q1")

    assert path is not None
    assert path.exists()
    assert path.suffix == ".wav"


def test_generate_should_reject_an_unknown_case(session):
    with pytest.raises(KeyError, match="nope"):
        session.generate("nope")


def test_talk_should_not_reset(session):
    """The Talk tab is a conversation: resetting each turn would destroy it."""
    session.talk(text="hello")
    session.talk(text="and then?")

    assert session._source.resets == 0


def test_pending_should_shrink_as_cases_are_rated(session):
    assert session.pending() == ["q1", "q2"]

    session.record("q1", intelligibility=4, naturalness=4, overall=4)

    assert session.pending() == ["q2"]


def test_pending_should_ignore_another_version(tmp_path):
    """Ratings from another checkpoint must not mark this one as done."""
    store = RatingStore(tmp_path / "r.jsonl")
    other = BenchSession(FakeSource(label="other"), _questions(), store=store)
    other.record("q1", intelligibility=3, naturalness=3, overall=3)

    mine = BenchSession(FakeSource(label="mine"), _questions(), store=store)

    assert mine.pending() == ["q1", "q2"]


def test_progress_should_report_against_the_version(session):
    session.record("q1", intelligibility=5, naturalness=5, overall=5)

    assert session.progress() == "1/2 rated for fake@test"


def test_keeps_history_should_follow_the_source(tmp_path):
    stateless = BenchSession(FakeSource(stateful=False), _questions(), store=RatingStore(tmp_path / "r.jsonl"))

    assert not stateless.keeps_history
