"""Campaign comparison (`lfm2_audio.evaluation.comparison`) — step 7 decision aid."""

from __future__ import annotations

import json
from typing import Any

from lfm2_audio.evaluation.comparison import CampaignComparison, MetricDelta, compare_files, compare_reports


def _report(metrics: list[dict[str, Any]], **context: Any) -> dict[str, Any]:
    return {"context": context, "cases": context.pop("cases", 24), "metrics": metrics}


def _metric(scorer: str, mean: float | None, higher: bool) -> dict[str, Any]:
    return {"scorer": scorer, "mean": mean, "higher_is_better": higher}


class TestMetricDelta:
    def test_should_treat_a_decrease_as_progress_when_lower_is_better(self) -> None:
        delta = MetricDelta(scorer="wer", higher_is_better=False, baseline=0.30, candidate=0.20)

        assert delta.improved is True
        assert delta.relative_pct == -33.3

    def test_should_treat_an_increase_as_progress_when_higher_is_better(self) -> None:
        delta = MetricDelta(scorer="dnsmos", higher_is_better=True, baseline=3.0, candidate=3.6)

        assert delta.improved is True
        assert delta.relative_pct == 20.0

    def test_should_flag_a_regression(self) -> None:
        delta = MetricDelta(scorer="dnsmos", higher_is_better=True, baseline=3.6, candidate=3.0)

        assert delta.improved is False

    def test_should_not_call_an_unmeasured_metric_a_tie(self) -> None:
        delta = MetricDelta(scorer="nisqa", higher_is_better=True, baseline=None, candidate=4.1)

        assert delta.improved is None
        assert delta.delta is None
        assert "—" in delta.as_row()

    def test_should_report_no_change_as_not_improved(self) -> None:
        delta = MetricDelta(scorer="wer", higher_is_better=False, baseline=0.25, candidate=0.25)

        assert delta.improved is False

    def test_should_skip_relative_pct_when_baseline_is_zero(self) -> None:
        delta = MetricDelta(scorer="wer", higher_is_better=False, baseline=0.0, candidate=0.1)

        assert delta.relative_pct is None


class TestCompareReports:
    def test_should_pair_metrics_and_detect_a_clean_win(self) -> None:
        baseline = _report([_metric("wer", 0.30, False), _metric("dnsmos", 3.0, True)])
        candidate = _report([_metric("wer", 0.22, False), _metric("dnsmos", 3.4, True)])

        comparison = compare_reports(baseline, candidate)

        assert len(comparison.deltas) == 2
        assert comparison.is_win is True
        assert not comparison.regressed

    def test_should_not_call_a_mixed_result_a_win(self) -> None:
        baseline = _report([_metric("wer", 0.30, False), _metric("dnsmos", 3.4, True)])
        candidate = _report([_metric("wer", 0.22, False), _metric("dnsmos", 3.0, True)])

        comparison = compare_reports(baseline, candidate)

        assert comparison.is_win is False
        assert [d.scorer for d in comparison.regressed] == ["dnsmos"]

    def test_should_warn_when_question_sets_differ(self) -> None:
        baseline = _report([_metric("wer", 0.3, False)], questions="a.jsonl", max_tokens=400)
        candidate = _report([_metric("wer", 0.2, False)], questions="b.jsonl", max_tokens=400)

        comparison = compare_reports(baseline, candidate)

        assert any("questions" in w for w in comparison.warnings)

    def test_should_warn_when_sample_counts_differ(self) -> None:
        baseline = {"context": {}, "cases": 24, "metrics": []}
        candidate = {"context": {}, "cases": 12, "metrics": []}

        comparison = compare_reports(baseline, candidate)

        assert any("sample counts" in w for w in comparison.warnings)

    def test_should_keep_a_metric_present_on_one_side_only(self) -> None:
        comparison = compare_reports(_report([]), _report([_metric("nisqa", 4.0, True)]))

        assert [d.scorer for d in comparison.deltas] == ["nisqa"]
        assert comparison.deltas[0].baseline is None


class TestRendering:
    def test_should_render_a_markdown_table_with_the_verdict(self) -> None:
        comparison = CampaignComparison(
            deltas=(MetricDelta("wer", False, 0.30, 0.20),),
            warnings=("context mismatch on 'max_tokens': 400 vs 200",),
        )

        markdown = comparison.to_markdown(baseline_name="vanilla", candidate_name="ft-v1")

        assert "vanilla vs ft-v1" in markdown
        assert "| wer | 0.3000 | 0.2000 | improved (-33.3%) |" in markdown
        assert "1 improved, 0 regressed" in markdown
        assert "max_tokens" in markdown

    def test_should_compare_two_files_on_disk(self, tmp_path) -> None:
        left, right = tmp_path / "a.json", tmp_path / "b.json"
        left.write_text(json.dumps(_report([_metric("wer", 0.3, False)])), encoding="utf-8")
        right.write_text(json.dumps(_report([_metric("wer", 0.1, False)])), encoding="utf-8")

        comparison = compare_files(left, right)

        assert comparison.is_win is True
