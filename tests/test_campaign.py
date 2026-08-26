"""Tests for ``evaluation.campaign.Campaign`` (no GPU, no model)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from lfm2_audio.ds.campaign_config import CampaignConfig
from lfm2_audio.evaluation.campaign import Campaign
from lfm2_audio.evaluation.question import Question
from lfm2_audio.scorer.sample import EvalSample


class _StubGenerator:
    """Answers instantly, and records which thread served the variant."""

    def __init__(self, label: str, *, delay: float = 0.0, fail: bool = False) -> None:
        self.label = label
        self._delay = delay
        self._fail = fail
        self.threads: set[int] = set()

    def generate(self, question: Question) -> EvalSample:
        if self._fail:
            raise RuntimeError(f"{self.label} exploded")
        self.threads.add(threading.get_ident())
        time.sleep(self._delay)
        return EvalSample(
            sample_id=question.question_id,
            prompt_text=question.text,
            predicted_text=f"answer from {self.label}",
            trajectory=[{"kind": "answer", "content": f"answer from {self.label}"}],
        )


@pytest.fixture
def questions(tmp_path: Path) -> Path:
    path = tmp_path / "cases.jsonl"
    lines = [
        json.dumps(
            {
                "id": f"case_{i}",
                "turns": [
                    {"role": "user", "text": f"question {i}"},
                    {"role": "assistant", "text": f"answer {i}"},
                ],
            }
        )
        for i in range(3)
    ]
    path.write_text("\n".join(lines))
    return path


def _config(questions: Path, tmp_path: Path, *, names: list[str], parallel: int = 1) -> CampaignConfig:
    return CampaignConfig(
        questions=str(questions),
        runs_root=str(tmp_path / "runs"),
        max_parallel=parallel,
        scoring={"scorers": [{"name": "tool_call"}]},  # type: ignore[arg-type]
        variants=tuple({"name": name, "checkpoint": "stub"} for name in names),  # type: ignore[arg-type]
    )


def test_should_produce_one_run_per_variant(questions: Path, tmp_path: Path) -> None:
    config = _config(questions, tmp_path, names=["a", "b"])

    outcomes = Campaign(config, lambda variant: _StubGenerator(variant.name)).run()

    assert [o.name for o in outcomes] == ["a", "b"]
    assert all(o.succeeded for o in outcomes)
    for name in ("a", "b"):
        assert (Path(config.runs_root) / name / "report.json").exists()


def test_should_archive_the_trajectory_of_every_case(questions: Path, tmp_path: Path) -> None:
    """The steps are what a viewer replays; losing them makes a run unreadable."""
    config = _config(questions, tmp_path, names=["a"])

    outcome = Campaign(config, lambda variant: _StubGenerator(variant.name)).run()[0]

    archived = json.loads((outcome.archive / "case_0.json").read_text())
    assert archived["trajectory"][0]["kind"] == "answer"


def test_should_keep_the_other_variants_when_one_fails(questions: Path, tmp_path: Path) -> None:
    config = _config(questions, tmp_path, names=["ok", "broken"])

    def factory(variant: object) -> _StubGenerator:
        name = variant.name  # type: ignore[attr-defined]
        return _StubGenerator(name, fail=name == "broken")

    outcomes = Campaign(config, factory).run()  # type: ignore[arg-type]

    by_name = {o.name: o for o in outcomes}
    assert by_name["ok"].succeeded
    assert not by_name["broken"].succeeded
    assert "exploded" in by_name["broken"].error


def test_should_run_variants_concurrently_when_allowed(questions: Path, tmp_path: Path) -> None:
    """Two variants must not be serialised when max_parallel allows it."""
    config = _config(questions, tmp_path, names=["a", "b"], parallel=2)
    generators: list[_StubGenerator] = []

    def factory(variant: object) -> _StubGenerator:
        generator = _StubGenerator(variant.name, delay=0.05)  # type: ignore[attr-defined]
        generators.append(generator)
        return generator

    started = time.perf_counter()
    Campaign(config, factory).run()  # type: ignore[arg-type]
    elapsed = time.perf_counter() - started

    threads = {thread for generator in generators for thread in generator.threads}
    assert len(threads) == 2, "les deux variantes ont tourné sur le même thread"
    assert elapsed < 3 * 0.05 * 2, "les variantes ont été sérialisées"


def test_should_write_the_variant_identity_into_the_report(questions: Path, tmp_path: Path) -> None:
    config = _config(questions, tmp_path, names=["a"])

    outcome = Campaign(config, lambda variant: _StubGenerator(variant.name)).run()[0]

    assert outcome.report is not None
    context = outcome.report.as_dict()["context"]
    assert context["variant"] == "a"
    assert context["questions"] == str(questions)
