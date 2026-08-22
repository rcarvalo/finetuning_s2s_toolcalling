"""Tests du socle de callbacks d'entraînement (sans GPU)."""

from __future__ import annotations

from lfm2_audio.ds.training_config import TrainingConfig
from lfm2_audio.evaluation.pipeline import EvaluationPipeline
from lfm2_audio.evaluation.question import Question
from lfm2_audio.evaluation.question_set import QuestionSet
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.training.callback import TrainingCallback
from lfm2_audio.training.callback_builder import CallbackBuilder
from lfm2_audio.training.callback_list import CallbackList
from lfm2_audio.training.callbacks.scoring import ScoringCallback
from lfm2_audio.training.step_context import StepContext, perplexity


class RecordingCallback(TrainingCallback):
    def __init__(self, label: str = "rec") -> None:
        self._label = label
        self.events: list[str] = []

    @property
    def name(self) -> str:
        return self._label

    def on_train_begin(self, context: StepContext) -> None:
        self.events.append("begin")

    def on_step_end(self, context: StepContext) -> None:
        self.events.append(f"step-{context.step}")

    def on_train_end(self, context: StepContext) -> None:
        self.events.append("end")


class ExplodingCallback(TrainingCallback):
    def __init__(self) -> None:
        self.calls = 0

    def on_step_end(self, context: StepContext) -> None:
        self.calls += 1
        message = "boom"
        raise RuntimeError(message)


# --------------------------------------------------------------------------- #
# StepContext
# --------------------------------------------------------------------------- #


def test_every_should_fire_on_multiples_only():
    assert StepContext(step=100, max_steps=1000).every(50)
    assert not StepContext(step=101, max_steps=1000).every(50)


def test_every_should_never_fire_at_step_zero_or_zero_interval():
    assert not StepContext(step=0, max_steps=10).every(5)
    assert not StepContext(step=5, max_steps=10).every(0)


def test_progress_and_finality():
    context = StepContext(step=500, max_steps=1000)

    assert context.progress == 0.5
    assert not context.is_final
    assert StepContext(step=1000, max_steps=1000).is_final


def test_perplexity_should_be_capped():
    assert perplexity(0.0) == 1.0
    assert perplexity(1000.0) == perplexity(20.0)  # garde-fou overflow


# --------------------------------------------------------------------------- #
# CallbackList
# --------------------------------------------------------------------------- #


def test_should_broadcast_every_event():
    first, second = RecordingCallback("a"), RecordingCallback("b")
    callbacks = CallbackList([first, second])
    context = StepContext(step=1, max_steps=10)

    callbacks.on_train_begin(context)
    callbacks.on_step_end(context)
    callbacks.on_train_end(context)

    assert first.events == ["begin", "step-1", "end"] == second.events


def test_a_failing_callback_should_be_disabled_not_fatal():
    """Perdre le suivi wandb ne doit pas perdre l'entraînement."""
    exploding, healthy = ExplodingCallback(), RecordingCallback()
    callbacks = CallbackList([exploding, healthy])

    for step in (1, 2, 3):
        callbacks.on_step_end(StepContext(step=step, max_steps=10))

    assert exploding.calls == 1  # neutralisé après le premier échec
    assert healthy.events == ["step-1", "step-2", "step-3"]


def test_metrics_should_be_shared_across_callbacks_of_one_event():
    """Le contrat : les producteurs de métriques passent avant les publieurs."""

    class Producer(TrainingCallback):
        def on_step_end(self, context: StepContext) -> None:
            context.metrics["score/wer"] = 0.1

    class Publisher(TrainingCallback):
        def __init__(self) -> None:
            self.seen: dict[str, float] = {}

        def on_step_end(self, context: StepContext) -> None:
            self.seen = dict(context.metrics)

    publisher = Publisher()
    CallbackList([Producer(), publisher]).on_step_end(StepContext(step=1, max_steps=10))

    assert publisher.seen == {"score/wer": 0.1}


# --------------------------------------------------------------------------- #
# ScoringCallback
# --------------------------------------------------------------------------- #


class ConstantScorer(BaseScorer):
    name = "constant"

    def measure(self, sample: EvalSample) -> ScoreResult:
        return ScoreResult.ok(self.name, 0.75)


class EchoGenerator:
    def generate(self, question: Question) -> EvalSample:
        return EvalSample(sample_id=question.question_id, predicted_text="answer")


def _questions() -> QuestionSet:
    return QuestionSet(questions=(Question(question_id="q1", text="hi"),), source="unit")


def test_scoring_callback_should_publish_prefixed_metrics():
    callback = ScoringCallback(
        _questions(),
        EvaluationPipeline([ConstantScorer()]),
        interval=10,
        generator=EchoGenerator(),
    )
    context = StepContext(step=10, max_steps=100)

    callback.on_step_end(context)

    assert context.metrics == {"score/constant": 0.75}


def test_scoring_callback_should_stay_quiet_between_intervals():
    callback = ScoringCallback(
        _questions(), EvaluationPipeline([ConstantScorer()]), interval=10, generator=EchoGenerator()
    )
    context = StepContext(step=7, max_steps=100)

    callback.on_step_end(context)

    assert context.metrics == {}


def test_scoring_callback_should_measure_a_baseline_when_asked():
    callback = ScoringCallback(
        _questions(),
        EvaluationPipeline([ConstantScorer()]),
        interval=10,
        at_start=True,
        generator=EchoGenerator(),
    )
    context = StepContext(step=0, max_steps=100)

    callback.on_train_begin(context)

    assert context.metrics["score/constant"] == 0.75


# --------------------------------------------------------------------------- #
# CallbackBuilder
# --------------------------------------------------------------------------- #


def test_builder_should_mount_only_the_console_by_default():
    callbacks = CallbackBuilder(TrainingConfig(train_dataset="x")).build()

    assert [c.name for c in callbacks] == ["ConsoleCallback"]


def test_builder_should_put_the_producer_before_the_publisher():
    config = TrainingConfig(
        train_dataset="x",
        evaluation={
            "enabled": True,
            "question_set": "benchmark/toolcalling_en/cases.sample.jsonl",
            "max_questions": 2,
            "scoring": {"scorers": [{"name": "tool_call"}]},
        },
    )

    names = [c.name for c in CallbackBuilder(config).build()]

    assert names.index("ScoringCallback") < names.index("ConsoleCallback")


def test_builder_should_ignore_an_evaluation_without_question_set():
    config = TrainingConfig(train_dataset="x", evaluation={"enabled": True})

    assert [c.name for c in CallbackBuilder(config).build()] == ["ConsoleCallback"]
