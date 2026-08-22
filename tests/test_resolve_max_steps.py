"""`resolve_max_steps` — epochs become steps only when the corpus size is knowable."""

from __future__ import annotations

import pytest

from lfm2_audio.core.errors import TrainingConfigError
from lfm2_audio.ds.training_config import TrainingConfig
from lfm2_audio.training.step_budget import resolve_max_steps


class _Sized:
    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size


class _Unsized:
    """A loader that only streams — no length available."""


def test_should_use_max_steps_when_no_epoch_is_requested() -> None:
    assert resolve_max_steps(TrainingConfig(max_steps=900), _Sized(2729)) == 900


def test_should_convert_epochs_into_steps() -> None:
    assert resolve_max_steps(TrainingConfig(num_epochs=3, batch_size=8), _Sized(2729)) == 1023


def test_should_raise_a_typed_error_when_the_loader_has_no_length() -> None:
    with pytest.raises(TrainingConfigError, match="does not expose a length"):
        resolve_max_steps(TrainingConfig(num_epochs=2), _Unsized())


def test_should_not_touch_an_unsized_loader_without_epochs() -> None:
    """max_steps recipes must keep working on streaming loaders."""
    assert resolve_max_steps(TrainingConfig(max_steps=500), _Unsized()) == 500
