"""Aggregate a VERSA pass into the numbers a gate is decided on.

Two choices here are not cosmetic.

**Medians, not means.** The model degenerates into repetition loops on a few
percent of samples, in both languages. A mean over such a distribution mostly
reports how often it looped; a gate on speech quality wants the quality when it
does not.

**Split by the language actually spoken, not the language asked.** A model that
does not yet mirror answers French questions in English, and averaging both
together hides exactly the regression the bilingual gates exist to catch — the
0B baseline showed FR speech sitting 0.14 UTMOS under EN while the pooled mean
looked fine.
"""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from lfm2_audio.evaluation.eval_log_audio import LoggedReply
from lfm2_audio.scorer.text.lang_match import detect_language

METRIC_KEYS = {"dnsmos": "dns_overall", "utmos": "utmos", "nisqa": "nisqa_mos_pred"}
"""Our metric names → the keys VERSA writes in its JSONL."""

UNKNOWN_LANGUAGE = "?"


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """One metric over one group of replies."""

    n: int
    median: float | None
    mean: float | None

    @classmethod
    def of(cls, values: list[float]) -> MetricSummary:
        if not values:
            return cls(n=0, median=None, mean=None)
        return cls(n=len(values), median=round(statistics.median(values), 3), mean=round(statistics.mean(values), 3))


@dataclass
class VersaGateReport:
    """VERSA metrics per spoken language, plus the pooled figures."""

    by_language: dict[str, dict[str, MetricSummary]] = field(default_factory=dict)
    pooled: dict[str, MetricSummary] = field(default_factory=dict)
    languages_seen: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "languages_seen": self.languages_seen,
            "pooled": {name: dataclasses.asdict(summary) for name, summary in self.pooled.items()},
            "by_language": {
                lang: {name: dataclasses.asdict(summary) for name, summary in metrics.items()}
                for lang, metrics in self.by_language.items()
            },
        }

    def markdown(self) -> str:
        header = "| langue | n | DNSMOS méd. | UTMOS méd. | NISQA méd. |"
        lines = [header, "|---|---|---|---|---|"]
        for lang in sorted(self.by_language):
            metrics = self.by_language[lang]
            counts = metrics["utmos"].n
            cells = " | ".join(_cell(metrics.get(name)) for name in ("dnsmos", "utmos", "nisqa"))
            lines.append(f"| {lang} | {counts} | {cells} |")
        pooled = " | ".join(_cell(self.pooled.get(name)) for name in ("dnsmos", "utmos", "nisqa"))
        lines.append(f"| **toutes** | {self.pooled['utmos'].n if 'utmos' in self.pooled else 0} | {pooled} |")
        return "\n".join(lines)


def build_report(replies: Iterable[LoggedReply], scores: Mapping[str, Mapping[str, Any]]) -> VersaGateReport:
    """Combine the extracted replies with VERSA's per-utterance scores."""
    grouped: dict[str, dict[str, list[float]]] = {}
    pooled: dict[str, list[float]] = {name: [] for name in METRIC_KEYS}
    seen: dict[str, int] = {}

    for reply in replies:
        row = scores.get(reply.sample_id)
        if row is None:
            continue
        language = detect_language(reply.spoken_text) or UNKNOWN_LANGUAGE
        seen[language] = seen.get(language, 0) + 1
        bucket = grouped.setdefault(language, {name: [] for name in METRIC_KEYS})
        for name, key in METRIC_KEYS.items():
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bucket[name].append(float(value))
                pooled[name].append(float(value))

    return VersaGateReport(
        by_language={
            lang: {name: MetricSummary.of(values) for name, values in metrics.items()}
            for lang, metrics in grouped.items()
        },
        pooled={name: MetricSummary.of(values) for name, values in pooled.items()},
        languages_seen=seen,
    )


def _cell(summary: MetricSummary | None) -> str:
    return f"{summary.median:.2f}" if summary and summary.median is not None else "—"
