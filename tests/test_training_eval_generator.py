"""`TrainingEvalGenerator` — scoring must not leave the model in eval mode."""

from __future__ import annotations

import sys
import types

import pytest

from lfm2_audio.training.eval_generator import TrainingEvalGenerator


class _Model:
    def __init__(self) -> None:
        self.training = True
        self.transitions: list[str] = []

    def eval(self) -> None:
        self.training = False
        self.transitions.append("eval")

    def train(self) -> None:
        self.training = True
        self.transitions.append("train")


class _Inner:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.seen_training_state: bool | None = None
        self.model: _Model | None = None

    def generate(self, question: object) -> str:
        if self.model is not None:
            self.seen_training_state = self.model.training
        if self.fail:
            raise RuntimeError("generation blew up")
        return "sample"


@pytest.fixture(autouse=True)
def _stub_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = types.ModuleType("torch")

    class _InferenceMode:
        def __enter__(self) -> None: ...
        def __exit__(self, *exc: object) -> None: ...

    torch.inference_mode = _InferenceMode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_should_generate_in_eval_mode_then_restore_training() -> None:
    model, inner = _Model(), _Inner()
    inner.model = model

    result = TrainingEvalGenerator(inner, model).generate(question=object())  # type: ignore[arg-type]

    assert result == "sample"
    assert inner.seen_training_state is False, "generation must run with dropout off"
    assert model.training is True, "the next step must train normally"


def test_should_restore_training_mode_even_when_generation_fails() -> None:
    model = _Model()

    with pytest.raises(RuntimeError, match="blew up"):
        TrainingEvalGenerator(_Inner(fail=True), model).generate(question=object())  # type: ignore[arg-type]

    assert model.training is True


def test_should_not_flip_a_model_already_in_eval_mode_back_to_train() -> None:
    model = _Model()
    model.training = False

    TrainingEvalGenerator(_Inner(), model).generate(question=object())  # type: ignore[arg-type]

    assert model.training is False
