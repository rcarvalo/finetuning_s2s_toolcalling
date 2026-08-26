"""Tests for ``scorer.audio.utmos.UtmosScorer`` (model stubbed — no download)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.scorer.audio.utmos import UtmosScorer
from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.status import ScoreStatus

torch = pytest.importorskip("torch")


class _FakeUtmos:
    """Records the call and returns a fixed score, like the hub model."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, signal: Any, sample_rate: int) -> Any:
        self.calls.append(sample_rate)
        return torch.tensor([4.125])

    def to(self, _device: str) -> _FakeUtmos:
        return self

    def eval(self) -> _FakeUtmos:
        return self


@pytest.fixture
def scorer(monkeypatch: pytest.MonkeyPatch) -> tuple[UtmosScorer, _FakeUtmos]:
    fake = _FakeUtmos()
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: fake)
    return UtmosScorer(device="cpu"), fake


def _sample(sample_rate: int = 24_000) -> EvalSample:
    speech = np.zeros(sample_rate, dtype=np.float32)
    return EvalSample(sample_id="c1", predicted_audio=Waveform.of(speech, sample_rate))


def test_should_report_the_predicted_mos(scorer: tuple[UtmosScorer, _FakeUtmos]) -> None:
    result = scorer[0].score(_sample())

    assert result.status is ScoreStatus.OK
    assert result.value == pytest.approx(4.125)


def test_should_pass_the_native_sample_rate(scorer: tuple[UtmosScorer, _FakeUtmos]) -> None:
    """UTMOS resamples internally — resampling first would be a lossy no-op."""
    utmos, fake = scorer

    utmos.score(_sample(sample_rate=24_000))

    assert fake.calls == [24_000]


def test_should_reuse_the_loaded_model(scorer: tuple[UtmosScorer, _FakeUtmos]) -> None:
    utmos, fake = scorer

    utmos.score(_sample())
    utmos.score(_sample())

    assert len(fake.calls) == 2  # two scores, one model


def test_should_skip_a_sample_without_audio(scorer: tuple[UtmosScorer, _FakeUtmos]) -> None:
    result = scorer[0].score(EvalSample(sample_id="c2"))

    assert result.status is ScoreStatus.SKIPPED
    assert result.value is None
