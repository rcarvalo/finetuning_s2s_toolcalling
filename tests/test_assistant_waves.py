"""Deux vagues, une voix : l'env de chaque vague impose sa brique et ses familles."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

JOB = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "assistant_waves.py"


@pytest.fixture
def waves() -> Any:
    spec = importlib.util.spec_from_file_location("assistant_waves", JOB)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_should_send_french_families_to_brick_a_without_the_english_share(waves: Any) -> None:
    env = waves.wave_env({}, waves.WAVES[0])

    assert env["BRICK_A_HF_PATH"] == "A_assistant_speech"
    assert env["BRICK_A_SKIP_KINDS"] == "en"
    assert "tc_fr_v1.jsonl" in env["BRICK_A_SOURCES"] and "dialogues_v2.jsonl" in env["BRICK_A_SOURCES"]


def test_should_send_the_english_share_to_brick_d_with_the_same_voice(waves: Any) -> None:
    env = waves.wave_env({}, waves.WAVES[1])

    assert env["BRICK_A_HF_PATH"] == "D_english"
    assert env["BRICK_A_KINDS"] == "en"
    assert env["BRICK_A_VOICE"] == "fr_female"


def test_should_let_the_operator_override_defaults_but_never_the_wave(waves: Any) -> None:
    env = waves.wave_env({"BRICK_A_CONCURRENCY": "8", "BRICK_A_HF_PATH": "ailleurs"}, waves.WAVES[0])

    assert env["BRICK_A_CONCURRENCY"] == "8"
    assert env["BRICK_A_HF_PATH"] == "A_assistant_speech"
