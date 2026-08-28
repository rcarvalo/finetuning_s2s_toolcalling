"""The shipped plan must add up, and a plan that does not must fail loudly.

A corpus whose slices silently miss their announced total produces a training
mix nobody chose — the kind of defect that only surfaces as a disappointing gate
several GPU-hours later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lfm2_audio.data_prep.corpus_plan import CorpusPlan, CorpusPlanError, CorpusSlice

PLAN_PATH = Path(__file__).resolve().parents[1] / "configs/corpus/fr_150h.yaml"


@pytest.fixture
def plan() -> CorpusPlan:
    return CorpusPlan.from_yaml(PLAN_PATH)


def test_should_load_the_shipped_plan(plan: CorpusPlan) -> None:
    assert plan.name == "fr_150h"
    assert plan.summed_hours == pytest.approx(150.0, abs=0.5)


def test_should_hold_the_validated_80_20_ratio(plan: CorpusPlan) -> None:
    """80/20 FR/EN is the ratio the 125 h pilot validated, not a preference."""
    assert plan.fr_ratio == pytest.approx(plan.target_fr_ratio, abs=0.02)


def test_should_report_what_remains_to_synthesise(plan: CorpusPlan) -> None:
    """Stock counts: 26.4 h of dialogue corpus and the EN set are already there."""
    assert plan.to_produce_h < plan.summed_hours
    assert plan.to_produce_h == pytest.approx(103.6, abs=0.5)


def test_should_put_french_conversation_at_the_wer_plateau(plan: CorpusPlan) -> None:
    """The WER curve flattens near 100 h; the conversational registers must reach it."""
    conversational = sum(s.hours for s in plan.slices if s.lang == "fr" and s.register.startswith("dialogue"))
    assert conversational >= 90.0


def test_should_reject_a_plan_whose_slices_miss_the_total() -> None:
    with pytest.raises(CorpusPlanError, match="tranches"):
        CorpusPlan(
            name="cassé",
            total_hours=150.0,
            target_fr_ratio=0.8,
            slices=(CorpusSlice(name="x", brick="A", lang="fr", hours=10.0, register="dialogue", rationale="…"),),
        )


def test_should_reject_an_empty_plan() -> None:
    with pytest.raises(CorpusPlanError, match="sans tranche"):
        CorpusPlan(name="vide", total_hours=0.0, target_fr_ratio=0.8)
