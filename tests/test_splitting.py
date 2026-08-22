"""Stratified held-out split (`lfm2_audio.data_prep.splitting`)."""

from __future__ import annotations

import pytest

from lfm2_audio.data_prep.splitting import SplitReport, stratified_split, target_of


def _call(idx: int, tool: str) -> dict:
    return {
        "id": f"c{idx}",
        "turns": [
            {"role": "user", "text": f"utterance {idx}"},
            {"role": "assistant", "tool_calls": [{"name": tool, "arguments": {}}]},
        ],
    }


def _negative(idx: int) -> dict:
    return {"id": f"n{idx}", "turns": [{"role": "user", "text": f"chat {idx}"}, {"role": "assistant", "text": "hi"}]}


def _corpus() -> list[dict]:
    # 500 web_search / 300 db_query / 200 negatives
    return (
        [_call(i, "web_search") for i in range(500)]
        + [_call(1000 + i, "db_query") for i in range(300)]
        + [_negative(i) for i in range(200)]
    )


class TestTargetOf:
    def test_should_read_the_tool_name_from_the_assistant_turn(self) -> None:
        assert target_of(_call(1, "web_search")) == "web_search"

    def test_should_return_none_when_no_tool_is_called(self) -> None:
        assert target_of(_negative(1)) == "none"

    def test_should_ignore_meta_and_trust_the_turn(self) -> None:
        dialogue = _negative(1) | {"meta": {"target": "web_search"}}

        assert target_of(dialogue) == "none"


class TestStratifiedSplit:
    def test_should_return_exactly_the_requested_test_size(self) -> None:
        _, test, _ = stratified_split(_corpus(), test_size=200)

        assert len(test) == 200

    def test_should_preserve_the_target_distribution(self) -> None:
        _, _, report = stratified_split(_corpus(), test_size=200)

        # source shares are 50 / 30 / 20 percent
        assert report.test_targets["web_search"] == pytest.approx(100, abs=3)
        assert report.test_targets["db_query"] == pytest.approx(60, abs=3)
        assert report.test_targets["none"] == pytest.approx(40, abs=3)

    def test_should_not_leave_a_dialogue_in_both_sides(self) -> None:
        train, test, _ = stratified_split(_corpus(), test_size=150)

        assert not {d["id"] for d in train} & {d["id"] for d in test}
        assert len(train) + len(test) == 1000

    def test_should_be_deterministic_for_a_given_seed(self) -> None:
        first = stratified_split(_corpus(), test_size=100, seed=7)[1]
        second = stratified_split(_corpus(), test_size=100, seed=7)[1]

        assert [d["id"] for d in first] == [d["id"] for d in second]

    def test_should_change_with_the_seed(self) -> None:
        first = stratified_split(_corpus(), test_size=100, seed=1)[1]
        second = stratified_split(_corpus(), test_size=100, seed=2)[1]

        assert [d["id"] for d in first] != [d["id"] for d in second]

    def test_should_refuse_a_test_size_larger_than_the_corpus(self) -> None:
        with pytest.raises(ValueError, match="fewer than"):
            stratified_split(_corpus(), test_size=1000)

    def test_should_refuse_a_non_positive_test_size(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            stratified_split(_corpus(), test_size=0)


class TestSplitReport:
    def test_should_summarize_the_test_composition_in_percent(self) -> None:
        report = SplitReport(train=800, test=200, train_targets={}, test_targets={"web_search": 100, "none": 100})

        summary = report.summary()

        assert "train 800 / test 200" in summary
        assert "web_search=100 (50%)" in summary
