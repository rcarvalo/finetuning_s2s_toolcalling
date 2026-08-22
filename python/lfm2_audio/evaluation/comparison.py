"""Compare two evaluation campaigns — the deliverable of the last weekend step.

A fine-tune is only worth keeping if it moves the metrics in the right
direction, and "the right direction" differs per metric (WER down, MOS up).
:class:`MetricDelta` reads ``higher_is_better`` from the report instead of
hard-coding a list of metric names.

Two campaigns are only comparable when their ``context`` blocks agree on the
question set and the generation settings: :func:`compare_reports` refuses to
compare silently and surfaces the mismatch as a warning in the result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMPARABILITY_KEYS = ("questions", "cases", "max_tokens", "asr_model_id")
"""Context entries that must match for a comparison to mean anything."""


def _fmt(value: float | None) -> str:
    """Render a metric value, or an em dash when it was never measured."""
    return "—" if value is None else f"{value:.4f}"


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """One metric, before and after."""

    scorer: str
    higher_is_better: bool
    baseline: float | None
    candidate: float | None

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline

    @property
    def improved(self) -> bool | None:
        """``None`` when either side is missing — an unmeasured metric is not a tie."""
        change = self.delta
        if change is None or change == 0.0:
            return None if change is None else False
        return change > 0 if self.higher_is_better else change < 0

    @property
    def relative_pct(self) -> float | None:
        if self.delta is None or not self.baseline:
            return None
        return round(100.0 * self.delta / abs(self.baseline), 1)

    def as_row(self) -> str:
        arrow = {True: "improved", False: "regressed", None: "n/a"}[self.improved]
        pct = "" if self.relative_pct is None else f" ({self.relative_pct:+.1f}%)"
        return f"| {self.scorer} | {_fmt(self.baseline)} | {_fmt(self.candidate)} | {arrow}{pct} |"


@dataclass(frozen=True, slots=True)
class CampaignComparison:
    """Baseline vs candidate over every metric they share."""

    deltas: tuple[MetricDelta, ...]
    warnings: tuple[str, ...] = ()

    @property
    def improved(self) -> tuple[MetricDelta, ...]:
        return tuple(d for d in self.deltas if d.improved is True)

    @property
    def regressed(self) -> tuple[MetricDelta, ...]:
        return tuple(d for d in self.deltas if d.improved is False)

    @property
    def is_win(self) -> bool:
        """A candidate wins when something improved and nothing regressed."""
        return bool(self.improved) and not self.regressed

    def to_markdown(self, *, baseline_name: str = "baseline", candidate_name: str = "candidate") -> str:
        lines = [
            f"# Comparison — {baseline_name} vs {candidate_name}",
            "",
            "| Metric | Baseline | Candidate | Change |",
            "|---|---:|---:|---|",
            *(d.as_row() for d in self.deltas),
            "",
            f"**Verdict**: {len(self.improved)} improved, {len(self.regressed)} regressed.",
        ]
        if self.warnings:
            lines += ["", "## Warnings", *(f"- {w}" for w in self.warnings)]
        return "\n".join(lines) + "\n"


def _means(report: dict[str, Any]) -> dict[str, tuple[float | None, bool]]:
    return {m["scorer"]: (m.get("mean"), bool(m.get("higher_is_better", True))) for m in report.get("metrics", [])}


def _comparability_warnings(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    left, right = baseline.get("context", {}), candidate.get("context", {})
    warnings = [
        f"context mismatch on {key!r}: {left.get(key)!r} vs {right.get(key)!r}"
        for key in COMPARABILITY_KEYS
        if key in left or key in right
        if left.get(key) != right.get(key)
    ]
    if baseline.get("cases") != candidate.get("cases"):
        warnings.append(f"different sample counts: {baseline.get('cases')} vs {candidate.get('cases')}")
    return warnings


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> CampaignComparison:
    """Build the metric-by-metric comparison of two report payloads."""
    left, right = _means(baseline), _means(candidate)
    deltas = tuple(
        MetricDelta(
            scorer=scorer,
            higher_is_better=left.get(scorer, (None, True))[1],
            baseline=left.get(scorer, (None, True))[0],
            candidate=right.get(scorer, (None, True))[0],
        )
        for scorer in sorted(set(left) | set(right))
    )
    return CampaignComparison(deltas=deltas, warnings=tuple(_comparability_warnings(baseline, candidate)))


def compare_files(baseline_path: str | Path, candidate_path: str | Path) -> CampaignComparison:
    """Load two JSON reports from disk and compare them."""

    def load(path: str | Path) -> dict[str, Any]:
        return dict(json.loads(Path(path).read_text(encoding="utf-8")))

    return compare_reports(load(baseline_path), load(candidate_path))
