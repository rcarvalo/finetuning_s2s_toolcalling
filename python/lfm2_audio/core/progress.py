"""Timestamped step logging for long remote jobs.

Written after a Voxtral run was killed at 45 minutes of total silence. The job
was installing a multi-gigabyte stack with ``pip -q``, so there was no way to
tell "working" from "hung" — and RunPod's own CPU/GPU metrics are no help: a pod
actively downloading at 7 files/s reports 2 % CPU and 0 % GPU, exactly what an
idle one reports.

So the job must say where it is, by itself, on a clock. Two rules:

* **every phase is announced before it starts**, with elapsed time — silence
  after a ``▶`` line then means "still inside that phase", which is
  actionable, unlike silence after nothing;
* **nothing that can take minutes runs quiet.** ``pip -q`` saves log lines that
  cost far more in GPU-hours than they ever saved in scrolling.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from types import TracebackType

Clock = Callable[[], float]
"""Injected so elapsed times are deterministic under test."""


class Progress:
    """Announces phases with elapsed time, on a stream that is never buffered."""

    def __init__(self, job: str, *, clock: Clock = time.monotonic) -> None:
        self._job = job
        self._clock = clock
        self._start = self._now()
        self._phase: str | None = None
        self._phase_start = self._start

    def _now(self) -> float:
        return self._clock()

    @property
    def elapsed(self) -> float:
        return self._now() - self._start

    def step(self, phase: str) -> None:
        """Close the running phase, announce the next one."""
        if self._phase is not None:
            self._emit(f"✓ {self._phase} — {self._now() - self._phase_start:.0f}s")
        self._phase = phase
        self._phase_start = self._now()
        self._emit(f"▶ {phase}")

    def note(self, message: str) -> None:
        """A heartbeat inside the current phase."""
        self._emit(f"  {message}")

    def done(self) -> None:
        if self._phase is not None:
            self._emit(f"✓ {self._phase} — {self._now() - self._phase_start:.0f}s")
            self._phase = None
        self._emit(f"■ {self._job} — total {self.elapsed:.0f}s")

    def _emit(self, body: str) -> None:
        print(f"[{self.elapsed:7.1f}s] {body}", flush=True)

    def __enter__(self) -> Progress:
        self._emit(f"■ {self._job} — démarrage")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc is not None and self._phase is not None:
            self._emit(f"✗ {self._phase} — ÉCHEC après {self._now() - self._phase_start:.0f}s : {exc}")
            self._phase = None
        self.done()


def stream_command(command: list[str], progress: Progress, *, every: int = 25) -> int:
    """Run a command, forwarding a line every ``every`` so silence means stuck.

    Full output would drown the log (pip prints a line per wheel); none at all is
    what cost the 45 minutes. One line in twenty-five keeps the log readable and
    still moves visibly.
    """
    import subprocess

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for index, line in enumerate(process.stdout):
        if index % every == 0:
            progress.note(line.rstrip()[:160])
    return process.wait()
