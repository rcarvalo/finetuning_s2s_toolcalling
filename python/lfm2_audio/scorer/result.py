"""``ScoreResult`` — une mesure d'un scorer sur un échantillon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from lfm2_audio.scorer.status import ScoreStatus


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Résultat typé d'un scorer.

    ``value`` n'a de sens que si ``status`` vaut ``OK`` ; les constructeurs
    nommés ci-dessous rendent impossible de produire une valeur avec un statut
    d'échec, ou l'inverse.

    ``higher_is_better`` voyage avec le résultat : le WER se lit à l'envers du
    DNSMOS, et l'agrégation ne doit pas avoir à le deviner.
    """

    scorer: str
    status: ScoreStatus
    value: float | None = None
    higher_is_better: bool = True
    reason: str = ""
    """Pourquoi il n'y a pas de valeur (dépendance absente, erreur, hors périmètre)."""

    details: dict[str, Any] = field(default_factory=dict)
    """Sous-mesures et pièces à conviction (transcription, note par critère…)."""

    @classmethod
    def ok(
        cls,
        scorer: str,
        value: float,
        *,
        higher_is_better: bool = True,
        details: dict[str, Any] | None = None,
    ) -> Self:
        return cls(
            scorer=scorer,
            status=ScoreStatus.OK,
            value=value,
            higher_is_better=higher_is_better,
            details=details or {},
        )

    @classmethod
    def unavailable(cls, scorer: str, reason: str) -> Self:
        return cls(scorer=scorer, status=ScoreStatus.UNAVAILABLE, reason=reason)

    @classmethod
    def skipped(cls, scorer: str, reason: str) -> Self:
        return cls(scorer=scorer, status=ScoreStatus.SKIPPED, reason=reason)

    @classmethod
    def failed(cls, scorer: str, reason: str) -> Self:
        return cls(scorer=scorer, status=ScoreStatus.FAILED, reason=reason)

    @property
    def is_measurement(self) -> bool:
        return self.status.is_measurement and self.value is not None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"scorer": self.scorer, "status": str(self.status)}
        if self.value is not None:
            payload["value"] = self.value
        if self.reason:
            payload["reason"] = self.reason
        if self.details:
            payload["details"] = self.details
        return payload
