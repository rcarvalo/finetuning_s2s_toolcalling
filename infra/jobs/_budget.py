"""``Budget`` — one ledger for every shard of a job, whatever thread runs it.

Each generator run enforces its own ``--max-usd`` and reports ``===SPEND===``
when it ends; this ledger adds the reports up and decides whether the NEXT
shard may start, and with what allowance. Overshoot is bounded by one shard
per thread, never by the job.
"""

from __future__ import annotations

import re
import threading

SPEND_RE = re.compile(r"===SPEND===.*\busd=([0-9.]+)")


def parse_spend(line: str) -> float | None:
    """The dollars on a ``===SPEND===`` line, or None for any other line."""
    match = SPEND_RE.search(line)
    return float(match.group(1)) if match else None


class Budget:
    def __init__(self, max_usd: float | None) -> None:
        self._max_usd = max_usd
        self._spent = 0.0
        self._rows = 0
        self._exhausted = False
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def max_usd(self) -> float | None:
        return self._max_usd

    def record(self, usd: float, rows: int = 0) -> None:
        with self._lock:
            self._spent += usd
            self._rows += rows

    def exhaust(self) -> None:
        """A shard stopped on its own cap: nothing else starts."""
        with self._lock:
            self._exhausted = True

    def remaining(self) -> float | None:
        if self._max_usd is None:
            return None
        return max(0.0, self._max_usd - self._spent)

    def can_start(self) -> bool:
        remaining = self.remaining()
        return not self._exhausted and (remaining is None or remaining > 0)

    def allowance(self, parallel: int) -> float | None:
        """What ONE shard may spend, with ``parallel`` shards possibly in flight."""
        remaining = self.remaining()
        return None if remaining is None else remaining / max(1, parallel)

    def per_row(self) -> float | None:
        """Measured cost of one dialogue this run, or None before anything was paid."""
        return self._spent / self._rows if self._rows else None

    def summary(self, missing: int) -> str:
        """``===PROJECTION===``: what this run cost and what the rest would."""
        per_row = self.per_row()
        projected = "?" if per_row is None else f"{per_row * missing:.2f}"
        cap = "none" if self._max_usd is None else f"{self._max_usd:.2f}"
        return (
            f"===PROJECTION=== spent_usd={self._spent:.4f} cap_usd={cap} rows={self._rows} "
            f"per_dialogue_usd={'?' if per_row is None else f'{per_row:.5f}'} "
            f"missing={missing} projected_usd={projected}"
        )
