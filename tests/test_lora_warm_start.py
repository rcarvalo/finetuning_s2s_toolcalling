"""Warm start of a LoRA adapter (`LoraConfig.init_adapter`, `lora.warm_start_lora`).

Colab reclaims VMs mid-run, so a training that cannot resume from its last
pushed adapter can never finish a multi-hour recipe. These tests pin the two
halves of that path: the recipe field that carries the source, and the loader
that accepts a local directory without touching the Hub.

The loader lives behind `peft`, a training-only extra: its tests skip on a
plain install rather than dragging peft into the default test environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lfm2_audio.ds.training_config import TrainingConfig


class _FakeModel:
    """Minimal stand-in: only `load_state_dict` is exercised by the loader."""

    def __init__(self) -> None:
        self.loaded: dict[str, Any] = {}

    def load_state_dict(self, weights: dict[str, Any], strict: bool = True) -> tuple[list, list]:
        self.loaded = weights
        return [], []


def test_should_default_init_adapter_to_none() -> None:
    assert TrainingConfig().lora.init_adapter is None


def test_should_carry_the_adapter_source_from_the_recipe() -> None:
    config = TrainingConfig(lora={"init_adapter": "Rcarvalo/lfm25-tc-en-v3-adapter"})

    assert config.lora.init_adapter == "Rcarvalo/lfm25-tc-en-v3-adapter"


def test_should_load_weights_from_a_local_adapter_directory(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    import torch
    from safetensors.torch import save_file

    from lfm2_audio.training.lora import warm_start_lora

    save_file({"lfm.layers.0.q_proj.lora_A.weight": torch.ones(2, 2)}, str(tmp_path / "adapter_model.safetensors"))
    model = _FakeModel()

    warm_start_lora(model, str(tmp_path))

    assert torch.equal(model.loaded["lfm.layers.0.q_proj.lora_A.weight"], torch.ones(2, 2))


def test_should_not_fall_through_to_the_hub_for_an_empty_directory(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    from lfm2_audio.training.lora import warm_start_lora

    with pytest.raises(FileNotFoundError, match=r"adapter_model\.safetensors"):
        warm_start_lora(_FakeModel(), str(tmp_path))
