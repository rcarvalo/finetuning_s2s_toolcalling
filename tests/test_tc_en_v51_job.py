"""``derive_resume_config`` — la reprise à chaud de v4, appliquée à v5.1.

Une VM Colab meurt sans prévenir ; la seule chose qui survit est l'adaptateur
poussé sur le Hub. La reprise doit repartir de lui, finir les pas restants et
NE PAS rejouer le warmup — v4 l'a payé (`tc_en_voice_agent_v4_resume.yaml`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

JOB = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "tc_en_v51.py"


@pytest.fixture
def job() -> Any:
    spec = importlib.util.spec_from_file_location("tc_en_v51", JOB)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def base() -> dict[str, Any]:
    return {
        "lora": {"enabled": True, "r": 32},
        "num_epochs": 1.5,
        "warmup_steps": 150,
        "hub_repo": "Rcarvalo/lfm25-tc-en-v5_1-adapter",
        "wandb_run_name": "run",
    }


def test_should_warm_start_from_the_pushed_adapter(job: Any, base: dict[str, Any]) -> None:
    derived = job.derive_resume_config(base, 1000)

    assert derived["lora"]["init_adapter"] == "Rcarvalo/lfm25-tc-en-v5_1-adapter"
    assert derived["lora"]["r"] == 32


def test_should_finish_only_the_remaining_steps(job: Any, base: dict[str, Any]) -> None:
    derived = job.derive_resume_config(base, 1000)

    assert derived["max_steps"] == job.TOTAL_STEPS - 1000
    assert "num_epochs" not in derived


def test_should_not_replay_the_warmup(job: Any, base: dict[str, Any]) -> None:
    assert job.derive_resume_config(base, 1000)["warmup_steps"] == 30


def test_should_never_ask_for_zero_steps(job: Any, base: dict[str, Any]) -> None:
    assert job.derive_resume_config(base, 5000)["max_steps"] == 1


def test_should_leave_the_base_config_untouched(job: Any, base: dict[str, Any]) -> None:
    job.derive_resume_config(base, 1000)

    assert "init_adapter" not in base["lora"] and base["num_epochs"] == 1.5
