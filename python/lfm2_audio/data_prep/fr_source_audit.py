"""Comparative audit of candidate FR training sources (bilingual plan, phase 1.1).

The question the audit answers is the user's arbitration: which source supplies
which share of the FR corpus — decided on measured quality, not by decree. Each
source contributes a same-size clip sample, measured with the SAME metrics the
training gates use (VERSA MOS family), plus a label-cleanliness proxy: the WER
between the shipped transcript and an independent ASR's reading of the clip.

Pure aggregation logic — sampling and metric IO live in the CLI.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ClipAudit:
    """One sampled clip with whatever measurements were possible."""

    sample_id: str
    duration_s: float | None = None
    speaker: str = ""
    transcript: str = ""
    dnsmos: float | None = None
    utmos: float | None = None
    nisqa: float | None = None
    label_wer: float | None = None


@dataclass
class SourceAudit:
    """Aggregates for one source; renders as one row of the report."""

    name: str
    register: str = ""
    metadata_only: bool = False
    clips: list[ClipAudit] = field(default_factory=list)

    def add(self, clip: ClipAudit) -> None:
        self.clips.append(clip)

    @property
    def size(self) -> int:
        return len(self.clips)

    @property
    def speaker_count(self) -> int:
        return len({c.speaker for c in self.clips if c.speaker})

    def median(self, metric: str) -> float | None:
        values = [v for c in self.clips if (v := getattr(c, metric)) is not None]
        return statistics.median(values) if values else None

    def duration_p10_p90(self) -> tuple[float, float] | None:
        values = sorted(c.duration_s for c in self.clips if c.duration_s is not None)
        if len(values) < 10:
            return None
        return values[len(values) // 10], values[(len(values) * 9) // 10]

    def summary(self) -> dict[str, Any]:
        span = self.duration_p10_p90()
        return {
            "name": self.name,
            "register": self.register,
            "clips": self.size,
            "speakers": self.speaker_count,
            "duration_p10_p90": span,
            "median_duration_s": self.median("duration_s"),
            "median_dnsmos": self.median("dnsmos"),
            "median_utmos": self.median("utmos"),
            "median_nisqa": self.median("nisqa"),
            "median_label_wer": self.median("label_wer"),
        }


def _fmt(value: float | None, precision: int = 2) -> str:
    return f"{value:.{precision}f}" if value is not None else "—"


def audit_markdown(audits: list[SourceAudit]) -> str:
    """The comparison table for ``docs/fr_data_audit.md``."""
    lines = [
        "| source | registre | clips | locuteurs | durée méd. (s) | DNSMOS | UTMOS | NISQA | WER labels |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for audit in audits:
        s = audit.summary()
        note = " (métadonnées seules)" if audit.metadata_only else ""
        lines.append(
            f"| {s['name']}{note} | {s['register']} | {s['clips']} | {s['speakers']} "
            f"| {_fmt(s['median_duration_s'], 1)} | {_fmt(s['median_dnsmos'])} "
            f"| {_fmt(s['median_utmos'])} | {_fmt(s['median_nisqa'])} "
            f"| {_fmt(s['median_label_wer'])} |"
        )
    return "\n".join(lines)
