"""``reid`` et ``depth_distribution`` du job de dialogues FR par shards.

Chaque run du générateur numérote depuis zéro : sans ré-identification, deux
shards produisent les mêmes ids et la fusion en écrase la moitié. La
distribution des tours utilisateur est LE chiffre que le job doit faire bouger
(627/695 dialogues à deux tours).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

JOB = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "generate_dialogues_fr.py"


@pytest.fixture
def job() -> Any:
    spec = importlib.util.spec_from_file_location("generate_dialogues_fr", JOB)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_should_stamp_the_shard_into_every_id(job: Any) -> None:
    rows = [{"id": "c_deep_0012", "turns": []}, {"id": "c_deep_0013", "turns": []}]

    assert [r["id"] for r in job.reid(rows, "deep", 3)] == ["c_deep_s03_0012", "c_deep_s03_0013"]


def test_should_keep_two_shards_from_colliding(job: Any) -> None:
    rows = [{"id": "c_soc_0000", "turns": []}]

    first = job.reid(rows, "social", 0)[0]["id"]
    second = job.reid(rows, "social", 1)[0]["id"]

    assert first != second


def test_should_not_mutate_the_input_rows(job: Any) -> None:
    rows = [{"id": "c_en_0001", "turns": []}]

    job.reid(rows, "en", 2)

    assert rows[0]["id"] == "c_en_0001"


def test_should_count_user_turns_per_dialogue(job: Any) -> None:
    rows = [
        {"turns": [{"role": "user"}, {"role": "assistant"}, {"role": "user"}, {"role": "assistant"}]},
        {"turns": [{"role": "user"}, {"role": "assistant"}]},
        {"turns": [{"role": "user"}, {"role": "assistant"}] * 5},
    ]

    assert job.depth_distribution(rows) == {1: 1, 2: 1, 5: 1}
