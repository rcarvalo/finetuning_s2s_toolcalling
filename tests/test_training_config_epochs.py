"""Epoch-driven step count (`TrainingConfig.steps_for`).

A recipe reasons in passes over the data; the trainer counts steps. Deriving
one from the other in the config keeps "two epochs" meaningful when the corpus
grows, instead of a step count that silently means something else.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lfm2_audio.ds.training_config import TrainingConfig


def test_should_keep_max_steps_when_no_epoch_is_requested() -> None:
    assert TrainingConfig(max_steps=1500).steps_for(2729) == 1500


def test_should_derive_steps_from_epochs_and_batch_size() -> None:
    config = TrainingConfig(num_epochs=2, batch_size=8)

    assert config.steps_for(2729) == 682  # 2 * 2729 / 8


def test_should_scale_with_the_corpus() -> None:
    config = TrainingConfig(num_epochs=1, batch_size=4)

    assert config.steps_for(4000) == 2 * config.steps_for(2000)


def test_should_accept_a_fractional_epoch() -> None:
    assert TrainingConfig(num_epochs=0.5, batch_size=2).steps_for(1000) == 250


def test_should_never_return_zero_step_on_a_tiny_corpus() -> None:
    assert TrainingConfig(num_epochs=1, batch_size=64).steps_for(10) == 1


def test_should_reject_an_empty_dataset() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        TrainingConfig(num_epochs=1).steps_for(0)


def test_should_reject_a_non_positive_epoch_count() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(num_epochs=0)
