"""``NisqaScorer`` must refuse the job it cannot do, rather than fail at measure time.

The official ``nisqa.tar`` ships weights only — ``{args, model_state_dict}`` —
and the model class lives in a repo that is not on PyPI. Before 2026-08-26 the
scorer declared itself available whenever the file existed, then crashed on the
first sample with ``KeyError: 'model'``. That defect stayed invisible for weeks
because the weights were missing everywhere, so it never got that far.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lfm2_audio.scorer.audio.nisqa import NisqaScorer

torch = pytest.importorskip("torch")


@pytest.fixture
def weights_only_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "nisqa.tar"
    torch.save({"args": {"name": "NISQAv2"}, "model_state_dict": {"w": torch.zeros(2)}}, path)
    return path


def test_should_report_unavailable_when_the_checkpoint_has_no_architecture(
    weights_only_checkpoint: Path,
) -> None:
    reason = NisqaScorer(weights_only_checkpoint).unavailable_reason()

    assert reason is not None
    assert "architecture" in reason
    assert "utmos" in reason  # the message points at the usable alternative


def test_should_report_unavailable_when_the_checkpoint_is_missing(tmp_path: Path) -> None:
    reason = NisqaScorer(tmp_path / "absent.tar").unavailable_reason()

    assert reason is not None
    assert "introuvable" in reason


def test_should_accept_a_checkpoint_carrying_a_model(tmp_path: Path) -> None:
    path = tmp_path / "nisqa.tar"
    torch.save({"model": torch.nn.Linear(2, 1)}, path)

    assert NisqaScorer(path).unavailable_reason() is None
