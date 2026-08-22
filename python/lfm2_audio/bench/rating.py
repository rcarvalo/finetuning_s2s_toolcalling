"""``Rating`` — one human verdict on one generated answer.

Absolute 1-5 scores, deliberately: an A/B protocol is more stable against rater
drift, but absolute scores are what makes a run comparable to published MOS
figures and to what DNSMOS claims to predict. The three axes mirror DNSMOS's
own sub-scores so the two can be correlated case by case.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self

SCALE_MIN = 1
SCALE_MAX = 5

AXES = ("intelligibility", "naturalness", "overall")


@dataclass(frozen=True, slots=True)
class Rating:
    """A listener's judgement of one (case, model version) pair."""

    case_id: str
    version: str
    """Which model produced the audio — checkpoint plus backend."""

    intelligibility: int
    """Can the words be made out at all? Closest to DNSMOS `sig`."""

    naturalness: int
    """Does it sound like speech rather than a machine? Closest to DNSMOS `bak`."""

    overall: int
    """Would you ship this answer? Closest to DNSMOS `ovrl`."""

    derailed: bool = False
    """The clip collapsed into a loop or babble instead of speaking the text.

    Tracked apart from the scores because it is a different kind of failure: a
    derailed clip is not "low quality", it is a broken generation, and averaging
    it with ordinary scores hides how often it happens.
    """

    notes: str = ""
    rated_at: str = ""

    def __post_init__(self) -> None:
        for axis in AXES:
            value = getattr(self, axis)
            if not SCALE_MIN <= value <= SCALE_MAX:
                message = f"{axis} must be between {SCALE_MIN} and {SCALE_MAX}, got {value}"
                raise ValueError(message)

    @classmethod
    def create(
        cls,
        case_id: str,
        version: str,
        *,
        intelligibility: int,
        naturalness: int,
        overall: int,
        derailed: bool = False,
        notes: str = "",
    ) -> Self:
        """Build a rating stamped with the current time."""
        return cls(
            case_id=case_id,
            version=version,
            intelligibility=intelligibility,
            naturalness=naturalness,
            overall=overall,
            derailed=derailed,
            notes=notes,
            rated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )

    @property
    def mean_score(self) -> float:
        return float(sum(int(getattr(self, axis)) for axis in AXES)) / len(AXES)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "version": self.version,
            **{axis: getattr(self, axis) for axis in AXES},
            "derailed": self.derailed,
            "notes": self.notes,
            "rated_at": self.rated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        return cls(
            case_id=str(payload["case_id"]),
            version=str(payload["version"]),
            intelligibility=int(payload["intelligibility"]),
            naturalness=int(payload["naturalness"]),
            overall=int(payload["overall"]),
            derailed=bool(payload.get("derailed", False)),
            notes=str(payload.get("notes", "")),
            rated_at=str(payload.get("rated_at", "")),
        )
