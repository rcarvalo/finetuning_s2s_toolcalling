"""The clips a brick refused, with what the re-listen heard — kept, not lost.

Until 05/09 a rejected clip vanished: audio deleted, transcript discarded, only
a counter left. Half of a French wave went that way with no way to tell a
strict threshold from a broken voice. The log records every refusal and is
pushed with the manifest, so the acceptance rule can be judged on evidence
and a clip refused twice is not paid for a third time.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class RejectionLog:
    """Cumulative ``dropped.jsonl``: one row per refusal, several rows per clip allowed."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._rows: list[dict[str, object]] = []
        self._attempts: Counter[str] = Counter()

    def load(self, existing: Path | None) -> RejectionLog:
        """Start from the log already on the Hub (None or a missing file: from nothing)."""
        if existing is not None and existing.exists():
            for line in existing.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._remember(json.loads(line))
        self._write()
        return self

    def attempts(self, clip_id: str) -> int:
        return self._attempts[clip_id]

    def exhausted(self, max_attempts: int) -> int:
        """How many clips were refused at least ``max_attempts`` times."""
        return sum(1 for count in self._attempts.values() if count >= max_attempts)

    def record(self, clip_id: str, *, text: str, heard: str, wer: float, cer: float) -> None:
        self._remember({"id": clip_id, "text": text, "heard": heard, "wer": round(wer, 4), "cer": round(cer, 4)})
        self._write()

    def __len__(self) -> int:
        return len(self._rows)

    def _remember(self, row: dict[str, object]) -> None:
        self._rows.append(row)
        self._attempts[str(row["id"])] += 1

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            for row in self._rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
