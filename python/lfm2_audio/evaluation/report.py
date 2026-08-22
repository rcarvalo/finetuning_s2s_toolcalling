"""``EvaluationReport`` — résultat d'une campagne, lisible et sérialisable."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from lfm2_audio.scorer.aggregate import MetricSummary
from lfm2_audio.scorer.result import ScoreResult


@dataclass(frozen=True, slots=True)
class SampleReport:
    """Ce qu'un échantillon a donné, tous scorers confondus."""

    sample_id: str
    results: tuple[ScoreResult, ...]
    predicted_text: str = ""

    def value(self, scorer: str) -> float | None:
        for result in self.results:
            if result.scorer == scorer and result.is_measurement:
                return result.value
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "predicted_text": self.predicted_text,
            "scores": [r.as_dict() for r in self.results],
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Campagne complète : détail par échantillon et agrégats par métrique.

    ``context`` retient de quoi rendre la campagne reproductible — jeu de
    questions, checkpoint, backend, version de rubrique. Deux rapports sans ce
    contexte ne sont pas comparables, et rien ne le signale au moment de les
    comparer.
    """

    samples: tuple[SampleReport, ...] = ()
    summaries: tuple[MetricSummary, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_samples(cls, samples: list[SampleReport], *, context: dict[str, Any] | None = None) -> Self:
        """Agrège les résultats par scorer, en conservant l'ordre de déclaration."""
        by_scorer: dict[str, list[ScoreResult]] = {}
        for sample in samples:
            for result in sample.results:
                by_scorer.setdefault(result.scorer, []).append(result)

        summaries = tuple(MetricSummary.from_results(scorer, results) for scorer, results in by_scorer.items())
        return cls(samples=tuple(samples), summaries=summaries, context=context or {})

    def summary(self, scorer: str) -> MetricSummary | None:
        return next((s for s in self.summaries if s.scorer == scorer), None)

    @property
    def measured_metrics(self) -> tuple[MetricSummary, ...]:
        return tuple(s for s in self.summaries if s.is_measured)

    @property
    def unmeasured_metrics(self) -> tuple[MetricSummary, ...]:
        return tuple(s for s in self.summaries if not s.is_measured)

    def as_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "cases": len(self.samples),
            "metrics": [s.as_dict() for s in self.summaries],
        }

    def write_json(self, path: str | Path, *, with_samples: bool = True) -> Path:
        """Écrit le rapport. Le détail par échantillon est le seul moyen de
        remonter d'une moyenne décevante au cas qui l'explique."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.as_dict()
        if with_samples:
            payload["samples"] = [s.as_dict() for s in self.samples]
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return destination

    def render(self) -> str:
        """Tableau lisible en terminal."""
        if not self.summaries:
            return "(aucune métrique)"

        width = max(len(s.scorer) for s in self.summaries)
        lines = [f"{len(self.samples)} cas évalués", ""]
        for summary in self.summaries:
            if summary.is_measured and summary.mean is not None:
                arrow = "↑" if summary.higher_is_better else "↓"
                coverage = "" if summary.coverage == 1.0 else f"  ({summary.measured} cas)"
                lines.append(f"  {summary.scorer:<{width}}  {summary.mean:6.3f} {arrow}{coverage}")
            else:
                lines.append(f"  {summary.scorer:<{width}}       —   {summary.reason}")
        return "\n".join(lines)
