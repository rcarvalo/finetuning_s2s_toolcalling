"""Tests de la pipeline d'évaluation et du rapport (sans modèle réel)."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from lfm2_audio.evaluation.pipeline import EvaluationPipeline
from lfm2_audio.evaluation.question import Question
from lfm2_audio.evaluation.question_set import QuestionSet
from lfm2_audio.evaluation.report import EvaluationReport, SampleReport
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample

QUESTIONS = QuestionSet(
    questions=(
        Question(question_id="q1", text="one", expected_calls=[{"name": "f", "arguments": {}}]),
        Question(question_id="q2", text="two"),
    ),
    source="unit-test",
)


class FixedScorer(BaseScorer):
    name: ClassVar[str] = "fixed"

    def __init__(self, value: float = 1.0) -> None:
        self._value = value

    def measure(self, sample: EvalSample) -> ScoreResult:
        return ScoreResult.ok(self.name, self._value)


class BrokenScorer(BaseScorer):
    name: ClassVar[str] = "broken"

    def measure(self, sample: EvalSample) -> ScoreResult:
        message = "nope"
        raise RuntimeError(message)


class EchoGenerator:
    """Rend une réponse déterministe. Satisfait ``ResponseGenerator``."""

    def generate(self, question: Question) -> EvalSample:
        return EvalSample(
            sample_id=question.question_id,
            prompt_text=question.text,
            predicted_text=f"answer to {question.text}",
            expected_calls=question.expected_calls,
        )


class FailingGenerator:
    def generate(self, question: Question) -> EvalSample:
        message = "gpu on fire"
        raise RuntimeError(message)


# --------------------------------------------------------------------------- #


def test_should_score_every_question():
    report = EvaluationPipeline([FixedScorer(0.5)]).run(QUESTIONS, EchoGenerator())

    assert len(report.samples) == 2
    assert report.summary("fixed").mean == 0.5


def test_a_failing_generation_should_not_lose_the_campaign():
    report = EvaluationPipeline([FixedScorer()]).run(QUESTIONS, FailingGenerator())

    assert len(report.samples) == 2
    assert "generation_error" not in report.context  # l'échec est porté par l'échantillon


def test_a_failing_scorer_should_not_lose_the_campaign():
    report = EvaluationPipeline([FixedScorer(), BrokenScorer()]).run(QUESTIONS, EchoGenerator())

    assert report.summary("fixed").measured == 2
    assert report.summary("broken").failed == 2
    assert not report.summary("broken").is_measured


def test_context_should_record_the_question_set():
    """Comparer deux campagnes menées sur des jeux différents est l'erreur
    la plus facile à commettre : le contexte la rend visible."""
    report = EvaluationPipeline([FixedScorer()]).run(QUESTIONS, EchoGenerator())

    assert report.context["question_set"] == "unit-test"
    assert report.context["cases"] == 2
    assert report.context["positives"] == 1
    assert report.context["scorers"] == ["fixed"]


def test_score_should_work_on_pre_generated_samples():
    """Renoter sans refaire tourner le modèle : changer de rubrique coûte alors zéro GPU."""
    samples = [EvalSample(sample_id="s1", predicted_text="hi")]

    report = EvaluationPipeline([FixedScorer(0.25)]).score(samples)

    assert report.summary("fixed").mean == 0.25


# --------------------------------------------------------------------------- #
# Rapport
# --------------------------------------------------------------------------- #


def test_summary_should_count_each_status():
    results = [
        ScoreResult.ok("m", 1.0),
        ScoreResult.ok("m", 0.0),
        ScoreResult.skipped("m", "hors périmètre"),
        ScoreResult.unavailable("m", "poids absents"),
        ScoreResult.failed("m", "boom"),
    ]

    report = EvaluationReport.from_samples(
        [SampleReport(sample_id=f"s{i}", results=(r,)) for i, r in enumerate(results)]
    )
    summary = report.summary("m")

    assert summary.measured == 2
    assert summary.skipped == 1
    assert summary.unavailable == 1
    assert summary.failed == 1
    assert summary.mean == 0.5
    assert summary.coverage == pytest.approx(0.4)


def test_unmeasured_metric_should_carry_its_reason():
    report = EvaluationReport.from_samples(
        [SampleReport(sample_id="s1", results=(ScoreResult.unavailable("dnsmos", "poids absents"),))]
    )

    assert report.summary("dnsmos").reason == "poids absents"
    assert report.unmeasured_metrics[0].scorer == "dnsmos"


def test_render_should_mark_direction_and_missing_metrics():
    report = EvaluationReport.from_samples(
        [
            SampleReport(
                sample_id="s1",
                results=(
                    ScoreResult.ok("wer", 0.1, higher_is_better=False),
                    ScoreResult.unavailable("dnsmos", "poids absents"),
                ),
            )
        ]
    )

    rendered = report.render()

    assert "↓" in rendered  # le WER se lit à l'envers
    assert "poids absents" in rendered


def test_write_json_should_include_samples(tmp_path):
    report = EvaluationPipeline([FixedScorer()]).run(QUESTIONS, EchoGenerator())

    path = report.write_json(tmp_path / "nested" / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["cases"] == 2
    assert len(payload["samples"]) == 2
    assert payload["metrics"][0]["scorer"] == "fixed"


def test_write_json_can_omit_samples(tmp_path):
    report = EvaluationPipeline([FixedScorer()]).run(QUESTIONS, EchoGenerator())

    payload = json.loads(report.write_json(tmp_path / "r.json", with_samples=False).read_text(encoding="utf-8"))

    assert "samples" not in payload
