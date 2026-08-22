"""Tests de ``LayoutDetector`` — reconnaissance du layout d'un checkpoint."""

from __future__ import annotations

import json

import pytest

from lfm2_audio.core.errors import CheckpointError
from lfm2_audio.ds.checkpoint import Layout
from lfm2_audio.serving.checkpoint.detector import LayoutDetector

LIQUID_SECTIONS = {
    "lfm": {"hidden_size": 8},
    "encoder": {},
    "depthformer": {},
    "preprocessor": {"sample_rate": 16_000},
}


@pytest.fixture
def detector() -> LayoutDetector:
    return LayoutDetector()


def _checkpoint(tmp_path, config: dict, name: str = "ckpt"):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def test_should_detect_omni_by_architecture(detector, tmp_path):
    directory = _checkpoint(tmp_path, {"architectures": ["Lfm2AudioOmniModel"]})

    assert detector.detect(directory) is Layout.OMNI


def test_should_detect_omni_by_model_type(detector, tmp_path):
    directory = _checkpoint(tmp_path, {"model_type": "lfm2_audio"})

    assert detector.detect(directory) is Layout.OMNI


def test_should_detect_liquid_by_its_sections(detector, tmp_path):
    directory = _checkpoint(tmp_path, dict(LIQUID_SECTIONS))

    assert detector.detect(directory) is Layout.LIQUID


def test_should_detect_backbone_only_checkpoint(detector, tmp_path):
    directory = _checkpoint(tmp_path, {"architectures": ["Lfm2ForCausalLM"], "model_type": "lfm2"})

    assert detector.detect(directory) is Layout.BACKBONE


def test_should_detect_adapter_before_reading_any_config(detector, tmp_path):
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert detector.detect(directory) is Layout.ADAPTER


def test_omni_should_win_over_liquid_sections(detector, tmp_path):
    # Un checkpoint converti garde ses sections liquid : l'architecture tranche.
    config = {"architectures": ["Lfm2AudioOmniModel"], **LIQUID_SECTIONS}
    directory = _checkpoint(tmp_path, config)

    assert detector.detect(directory) is Layout.OMNI


def test_should_reject_a_directory_without_config(detector, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(CheckpointError, match="pas un checkpoint"):
        detector.detect(empty)


def test_should_reject_unparsable_config(detector, tmp_path):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "config.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(CheckpointError, match="illisible"):
        detector.detect(directory)


def test_should_reject_an_unrelated_model(detector, tmp_path):
    directory = _checkpoint(tmp_path, {"architectures": ["LlamaForCausalLM"], "model_type": "llama"})

    with pytest.raises(CheckpointError, match="layout non reconnu"):
        detector.detect(directory)


def test_should_read_the_base_model_declared_by_peft(detector, tmp_path):
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "LiquidAI/LFM2.5-Audio-1.5B"}), encoding="utf-8"
    )

    assert detector.base_model_of_adapter(directory) == "LiquidAI/LFM2.5-Audio-1.5B"


def test_should_reject_an_adapter_without_declared_base(detector, tmp_path):
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(CheckpointError, match="base_model_name_or_path"):
        detector.base_model_of_adapter(directory)
