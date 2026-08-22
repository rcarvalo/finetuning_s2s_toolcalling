"""``MetricSummary`` — agrégat d'un scorer sur une campagne."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Self

from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.status import ScoreStatus


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Ce qu'un scorer a donné sur l'ensemble des échantillons.

    Le compte par statut est conservé, pas seulement la moyenne : une métrique
    calculée sur 12 cas sur 300 n'a pas la même valeur qu'une métrique complète,
    et la moyenne seule ne le dit pas.
    """

    scorer: str
    higher_is_better: bool
    measured: int
    skipped: int
    unavailable: int
    failed: int
    mean: float | None = None
    median: float | None = None
    reason: str = ""
    """Raison dominante d'absence de mesure, quand il n'y en a aucune."""

    @classmethod
    def from_results(cls, scorer: str, results: list[ScoreResult]) -> Self:
        values = [r.value for r in results if r.is_measurement and r.value is not None]
        by_status = {status: [r for r in results if r.status is status] for status in ScoreStatus}
        higher = next((r.higher_is_better for r in results if r.is_measurement), True)

        reason = ""
        if not values:
            blocked = by_status[ScoreStatus.UNAVAILABLE] or by_status[ScoreStatus.FAILED]
            blocked = blocked or by_status[ScoreStatus.SKIPPED]
            reason = blocked[0].reason if blocked else "aucun échantillon"

        return cls(
            scorer=scorer,
            higher_is_better=higher,
            measured=len(values),
            skipped=len(by_status[ScoreStatus.SKIPPED]),
            unavailable=len(by_status[ScoreStatus.UNAVAILABLE]),
            failed=len(by_status[ScoreStatus.FAILED]),
            mean=statistics.fmean(values) if values else None,
            median=statistics.median(values) if values else None,
            reason=reason,
        )

    @property
    def coverage(self) -> float:
        """Part des échantillons réellement mesurés."""
        total = self.measured + self.skipped + self.unavailable + self.failed
        return self.measured / total if total else 0.0

    @property
    def is_measured(self) -> bool:
        return self.measured > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "scorer": self.scorer,
            "mean": self.mean,
            "median": self.median,
            "higher_is_better": self.higher_is_better,
            "measured": self.measured,
            "skipped": self.skipped,
            "unavailable": self.unavailable,
            "failed": self.failed,
            "coverage": self.coverage,
            **({"reason": self.reason} if self.reason else {}),
        }
