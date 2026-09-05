"""``Budget`` — le registre partagé des shards ; le job ne dépasse jamais d'un run entier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "_budget.py"


@pytest.fixture
def budget_module() -> Any:
    spec = importlib.util.spec_from_file_location("_budget", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestParseSpend:
    def test_should_read_the_dollars_of_a_spend_line(self, budget_module: Any) -> None:
        assert budget_module.parse_spend("===SPEND=== model=claude-sonnet-5 calls=30 in=1 out=2 usd=0.8125\n") == 0.8125

    def test_should_ignore_any_other_line(self, budget_module: Any) -> None:
        assert budget_module.parse_spend("  fr_deep 10/300 — accueil") is None


class TestBudget:
    def test_should_start_with_the_whole_cap(self, budget_module: Any) -> None:
        budget = budget_module.Budget(8.0)

        assert budget.remaining() == 8.0
        assert budget.can_start()

    def test_should_subtract_what_shards_report(self, budget_module: Any) -> None:
        budget = budget_module.Budget(8.0)

        budget.record(0.5, rows=300)
        budget.record(0.25, rows=100)

        assert budget.remaining() == pytest.approx(7.25)
        assert budget.spent == pytest.approx(0.75)
        assert budget.rows == 400

    def test_should_split_the_allowance_between_parallel_shards(self, budget_module: Any) -> None:
        budget = budget_module.Budget(8.0)

        assert budget.allowance(4) == 2.0

    def test_should_refuse_to_start_once_spent(self, budget_module: Any) -> None:
        budget = budget_module.Budget(1.0)
        budget.record(1.0)

        assert not budget.can_start()
        assert budget.remaining() == 0.0

    def test_should_refuse_to_start_after_a_shard_hit_its_own_cap(self, budget_module: Any) -> None:
        budget = budget_module.Budget(8.0)
        budget.exhaust()

        assert not budget.can_start()

    def test_should_never_limit_without_a_cap(self, budget_module: Any) -> None:
        budget = budget_module.Budget(None)
        budget.record(1000.0)

        assert budget.can_start()
        assert budget.allowance(4) is None

    def test_should_project_the_rest_from_the_measured_cost(self, budget_module: Any) -> None:
        budget = budget_module.Budget(8.0)
        budget.record(0.6, rows=300)  # 0,002 $ le dialogue

        line = budget.summary(missing=1000)

        assert "per_dialogue_usd=0.00200" in line
        assert "projected_usd=2.00" in line
        assert line.startswith("===PROJECTION=== spent_usd=0.6000 cap_usd=8.00 rows=300")

    def test_should_admit_it_cannot_project_before_paying(self, budget_module: Any) -> None:
        assert "projected_usd=?" in budget_module.Budget(8.0).summary(missing=10)
