"""Le journal des rejets : cumulatif, réécrit à chaque refus, compte les tentatives."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "_rejection_log.py"


@pytest.fixture
def rejection_log_cls() -> Any:
    spec = importlib.util.spec_from_file_location("_rejection_log", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RejectionLog


def test_should_start_empty_and_write_an_empty_file(rejection_log_cls: Any, tmp_path: Path) -> None:
    log = rejection_log_cls(tmp_path / "out" / "dropped.jsonl").load(None)

    assert len(log) == 0
    assert (tmp_path / "out" / "dropped.jsonl").read_text() == ""


def test_should_record_a_refusal_with_what_was_heard(rejection_log_cls: Any, tmp_path: Path) -> None:
    log = rejection_log_cls(tmp_path / "dropped.jsonl").load(None)

    log.record("c_01_t1", text="bonjour monsieur", heard="bonjour mon sieur", wer=0.5, cer=0.0625)

    row = json.loads((tmp_path / "dropped.jsonl").read_text().splitlines()[0])
    assert row == {"id": "c_01_t1", "text": "bonjour monsieur", "heard": "bonjour mon sieur", "wer": 0.5, "cer": 0.0625}
    assert log.attempts("c_01_t1") == 1


def test_should_resume_from_the_hub_log_and_count_attempts_across_runs(rejection_log_cls: Any, tmp_path: Path) -> None:
    hub = tmp_path / "hub_dropped.jsonl"
    hub.write_text(json.dumps({"id": "c_01_t1", "text": "x", "heard": "y", "wer": 1.0, "cer": 1.0}) + "\n")
    log = rejection_log_cls(tmp_path / "dropped.jsonl").load(hub)

    log.record("c_01_t1", text="x", heard="z", wer=1.0, cer=1.0)

    assert log.attempts("c_01_t1") == 2
    assert log.attempts("never_seen") == 0
    assert log.exhausted(2) == 1
    assert len((tmp_path / "dropped.jsonl").read_text().splitlines()) == 2


def test_should_ignore_a_missing_hub_log(rejection_log_cls: Any, tmp_path: Path) -> None:
    log = rejection_log_cls(tmp_path / "dropped.jsonl").load(tmp_path / "absent.jsonl")

    assert len(log) == 0
