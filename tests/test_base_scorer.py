"""Tests du contrat ``BaseScorer`` — la template method et ses garde-fous."""

from __future__ import annotations

from typing import ClassVar

from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.status import ScoreStatus


class ConstantScorer(BaseScorer):
    name: ClassVar[str] = "constant"

    def __init__(self, value: float = 1.0) -> None:
        self.calls = 0
        self._value = value

    def measure(self, sample: EvalSample) -> ScoreResult:
        self.calls += 1
        return ScoreResult.ok(self.name, self._value)


class ExplodingScorer(ConstantScorer):
    name: ClassVar[str] = "exploding"

    def measure(self, sample: EvalSample) -> ScoreResult:
        message = "boom"
        raise RuntimeError(message)


class UnavailableScorer(ConstantScorer):
    name: ClassVar[str] = "unavailable"

    def unavailable_reason(self) -> str | None:
        return "dépendance absente"


class NarrowScorer(ConstantScorer):
    name: ClassVar[str] = "narrow"

    def supports(self, sample: EvalSample) -> bool:
        return sample.has_predicted_audio

    def skip_reason(self, sample: EvalSample) -> str:
        return "pas d'audio"


def _sample() -> EvalSample:
    return EvalSample(sample_id="s1", predicted_text="hello")


def test_should_return_the_measurement():
    assert ConstantScorer(0.5).score(_sample()).value == 0.5


def test_should_convert_an_exception_into_a_failed_result():
    # Une campagne de 500 cas ne doit pas être perdue par un scorer capricieux.
    result = ExplodingScorer().score(_sample())

    assert result.status is ScoreStatus.FAILED
    assert "RuntimeError" in result.reason


def test_should_short_circuit_when_unavailable():
    scorer = UnavailableScorer()

    result = scorer.score(_sample())

    assert result.status is ScoreStatus.UNAVAILABLE
    assert scorer.calls == 0  # measure() n'est jamais atteint


def test_should_skip_out_of_scope_samples():
    scorer = NarrowScorer()

    result = scorer.score(_sample())

    assert result.status is ScoreStatus.SKIPPED
    assert result.reason == "pas d'audio"
    assert scorer.calls == 0


def test_is_available_should_follow_the_reason():
    assert ConstantScorer().is_available
    assert not UnavailableScorer().is_available


def test_score_many_should_note_every_sample():
    scorer = ConstantScorer()

    results = scorer.score_many([_sample(), _sample(), _sample()])

    assert len(results) == 3
    assert scorer.calls == 3
