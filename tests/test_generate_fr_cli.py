"""``lfm2-generate-fr`` — les familles planifiées d'avance, le plafond qui arrête proprement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from lfm2_audio.cli.data import generate_fr
from lfm2_audio.data_prep.synth_dialogues import ContaminationFilter
from lfm2_audio.scorer.text.llm_spend import SpendCapReachedError

DIALOGUE = json.dumps(
    [
        {
            "topic": "accueil",
            "turns": [
                {"role": "user", "text": "Bonjour, j'ai rendez-vous avec madame Martin."},
                {"role": "assistant", "text": "Bonjour, je la préviens tout de suite."},
            ],
        }
    ],
    ensure_ascii=False,
)


def _args(**overrides: int) -> argparse.Namespace:
    base = {"n_fr": 0, "n_switch": 0, "n_deep": 0, "n_social": 0, "n_en": 0, "per_call": 10}
    base.update(overrides)
    return argparse.Namespace(**base)


class TestPlanFamilies:
    def test_should_plan_one_prompt_per_call(self) -> None:
        families = generate_fr.plan_families(_args(n_deep=25))

        assert [(f.kind, f.target, len(f.prompts)) for f in families] == [("fr_deep", 25, 3)]

    def test_should_skip_families_nobody_asked_for(self) -> None:
        assert generate_fr.plan_families(_args()) == []

    def test_should_alternate_the_code_switch_direction(self) -> None:
        (family,) = generate_fr.plan_families(_args(n_switch=30))

        assert family.topics == ["français→anglais", "anglais→français", "français→anglais"]
        assert "français" in family.prompts[0] and "anglais" in family.prompts[1]

    def test_should_keep_the_historical_order_fr_switch_then_the_rest(self) -> None:
        families = generate_fr.plan_families(_args(n_fr=10, n_switch=10, n_deep=10, n_social=10, n_en=10))

        assert [f.kind for f in families] == ["fr", "code_switch", "fr_deep", "fr_social", "en"]


class _Judge:
    def __init__(self, replies: list[str], *, cap_after: int | None = None) -> None:
        self._replies = replies
        self._cap_after = cap_after
        self.calls = 0

    def judge(self, prompt: str) -> str:
        if self._cap_after is not None and self.calls >= self._cap_after:
            raise SpendCapReachedError("plafond 1.00 $ atteint")
        reply = self._replies[self.calls]
        self.calls += 1
        return reply


class TestRunFamily:
    def test_should_parse_filter_and_flush_every_batch(self) -> None:
        (family,) = generate_fr.plan_families(_args(n_deep=20))
        flushes: list[int] = []

        produced = generate_fr.run_family(
            _Judge([DIALOGUE, DIALOGUE]),
            family,
            ContaminationFilter(held_out=[]),
            flush=lambda rows: flushes.append(len(rows)),
        )

        assert len(produced) == 2
        assert flushes == [1, 2]
        assert produced[0].kind == "fr_deep"

    def test_should_ignore_a_batch_that_is_not_json(self) -> None:
        (family,) = generate_fr.plan_families(_args(n_deep=10))

        produced = generate_fr.run_family(
            _Judge(["désolé, pas de JSON"]), family, ContaminationFilter(held_out=[]), flush=lambda _: None
        )

        assert produced == []


class TestMain:
    def test_should_exit_3_and_keep_the_paid_batches_when_the_cap_is_reached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "deep.jsonl"
        monkeypatch.setattr(generate_fr, "make_judge", lambda *a, **k: _Judge([DIALOGUE], cap_after=1))
        monkeypatch.setattr(
            "sys.argv",
            ["lfm2-generate-fr", "--out", str(out), "--n-fr", "0", "--n-switch", "0", "--n-deep", "20", "--benchmarks"],
        )

        with pytest.raises(SystemExit) as stop:
            generate_fr.main()

        assert stop.value.code == generate_fr.EXIT_SPEND_CAP
        assert len(out.read_text(encoding="utf-8").splitlines()) == 1

    def test_should_exit_1_and_name_the_missing_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(
            "sys.argv", ["lfm2-generate-fr", "--out", str(tmp_path / "x.jsonl"), "--provider", "anthropic"]
        )

        with pytest.raises(SystemExit) as stop:
            generate_fr.main()

        assert stop.value.code == 1
