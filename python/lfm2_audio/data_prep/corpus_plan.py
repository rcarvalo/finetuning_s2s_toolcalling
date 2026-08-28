"""How the 150 h of synthetic corpus is split, and why each share is that size.

Every slice below is sized by a *measured* need, not by taste. The four anchors:

* **The hours buy intelligibility, not beauty.** Finetuning a codec TTS on target
  speech moves UTMOS little (10 h → 2.9, 50 h → 3.1, 100 h → 3.2 against 3.6 for
  real recordings) but drops WER by ~9 absolute points over that same range. Our
  R2 gate *is* a WER gate (round-trip < 20 %), and the 0B baseline measured FR
  speech at WER 0.527 and 1.7× too long. Hence the conversational slice sits at
  the ~100 h end of that curve rather than at 26 h.
* **Single-voice synthetic dialogue is the established recipe**, not a
  workaround: Moshi's instruct fine-tune used 20 000 h of TTS dialogue in one
  professional voice. We run the same shape three orders of magnitude smaller.
* **Filtering buys volume back.** A Wolof speech LM trained on 860 h filtered
  beat one trained on 65 000 h unfiltered. Every clip here is transcribed back
  and dropped on disagreement, which is what licenses the modest total.
* **Tool calling is counted in dialogues, not hours.** Our own English run
  reached 0.830 on fresh-300 with 2 729 dialogues; the French slice is sized to
  match that count, and its hours simply follow.

The FR/EN split is 80/20 — the ratio validated on the 125 h pilot (val_loss
2.02, English preserved). English is not training here, it is *insurance*: the
frozen anchors must not move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TOTAL_TOLERANCE_H = 0.5
"""Slices are authored by hand; a rounding drift is not a config error."""


class CorpusPlanError(Exception):
    """The plan does not describe a corpus that can be built."""


@dataclass(frozen=True, slots=True)
class CorpusSlice:
    """One share of the corpus, with the measurement that fixed its size."""

    name: str
    brick: str
    lang: str
    hours: float
    register: str
    rationale: str
    dialogues: int | None = None
    already_have_h: float = 0.0
    """Hours already in stock, so the plan states what remains to synthesise."""

    @property
    def to_produce_h(self) -> float:
        return max(0.0, round(self.hours - self.already_have_h, 2))


@dataclass(frozen=True, slots=True)
class CorpusPlan:
    """The whole corpus, checked against its own totals."""

    name: str
    total_hours: float
    target_fr_ratio: float
    slices: tuple[CorpusSlice, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.slices:
            raise CorpusPlanError(f"plan {self.name!r} sans tranche")
        drift = abs(self.summed_hours - self.total_hours)
        if drift > TOTAL_TOLERANCE_H:
            raise CorpusPlanError(
                f"plan {self.name!r} : les tranches font {self.summed_hours:.1f} h "
                f"pour un total annoncé de {self.total_hours:.1f} h"
            )

    @property
    def summed_hours(self) -> float:
        return round(sum(s.hours for s in self.slices), 2)

    def hours_for(self, lang: str) -> float:
        return round(sum(s.hours for s in self.slices if s.lang == lang), 2)

    @property
    def fr_ratio(self) -> float:
        return round(self.hours_for("fr") / self.summed_hours, 3)

    @property
    def to_produce_h(self) -> float:
        """What still has to be synthesised, once the stock is counted."""
        return round(sum(s.to_produce_h for s in self.slices), 2)

    def bricks(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for item in self.slices:
            totals[item.brick] = round(totals.get(item.brick, 0.0) + item.hours, 2)
        return totals

    @classmethod
    def from_yaml(cls, path: Path) -> CorpusPlan:
        """Load a plan; config lives in YAML, never hardcoded in a job."""
        payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            slices = tuple(CorpusSlice(**item) for item in payload["slices"])
            return cls(
                name=payload["name"],
                total_hours=float(payload["total_hours"]),
                target_fr_ratio=float(payload["target_fr_ratio"]),
                slices=slices,
            )
        except (KeyError, TypeError) as error:
            raise CorpusPlanError(f"plan illisible dans {path}: {error}") from error
