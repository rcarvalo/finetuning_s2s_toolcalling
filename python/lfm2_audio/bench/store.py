"""``RatingStore`` — append-only persistence for human verdicts.

JSONL because ratings accumulate across sessions and versions: appending never
rewrites what is already there, a half-finished session loses nothing, and the
file stays diffable in git next to the code that produced the audio.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from lfm2_audio.bench.rating import Rating

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("reports/human_ratings.jsonl")


class RatingStore:
    """Reads and appends :class:`Rating` records."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, rating: Rating) -> None:
        """Add one verdict. Creates the file and its parent on first use."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rating.as_dict(), ensure_ascii=False) + "\n")

    def load(self) -> list[Rating]:
        """Every verdict recorded so far. Malformed lines are skipped, not fatal."""
        if not self._path.exists():
            return []

        ratings: list[Rating] = []
        with self._path.open(encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    ratings.append(Rating.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError) as error:
                    logger.warning("%s:%d unreadable, skipped (%s)", self._path, line_no, error)
        return ratings

    def rated_cases(self, version: str) -> set[str]:
        """Cases already judged for a version — so a session can resume."""
        return {r.case_id for r in self.load() if r.version == version}

    def versions(self) -> list[str]:
        return sorted({r.version for r in self.load()})

    def summary(self, version: str) -> dict[str, float | int]:
        """Aggregate one version's verdicts.

        ``derailed`` is counted rather than averaged into the scores: a broken
        generation is a different failure from a merely poor one, and mixing
        them hides how often the model collapses.
        """
        ratings = [r for r in self.load() if r.version == version]
        if not ratings:
            return {"rated": 0}

        healthy = [r for r in ratings if not r.derailed]
        aggregate: dict[str, float | int] = {
            "rated": len(ratings),
            "derailed": sum(r.derailed for r in ratings),
        }
        for axis in ("intelligibility", "naturalness", "overall"):
            values = [getattr(r, axis) for r in healthy]
            aggregate[axis] = sum(values) / len(values) if values else 0.0
        return aggregate

    def by_case(self) -> dict[str, list[Rating]]:
        """Verdicts grouped by case — the shape a version comparison needs."""
        grouped: dict[str, list[Rating]] = defaultdict(list)
        for rating in self.load():
            grouped[rating.case_id].append(rating)
        return dict(grouped)
