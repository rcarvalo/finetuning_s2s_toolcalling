"""Make an event stream playable without letting the listener throttle the model.

Two distinct failures show up when agent events are streamed straight to a
browser, and they need two distinct fixes:

*Backpressure.* Gradio blocks the generator on every ``yield`` until the
transport accepts the message — about 200 ms over a ``gradio.live`` tunnel.
Yielding raw Mimi frames (1920 samples, 80 ms) therefore paces generation at
network speed: measured 30 s to produce 11 s of speech, against ~7 s with no
streaming at all. :func:`detach` drains the source in a thread so the model
runs at its own pace whatever the listener does.

*Starvation.* An 80 ms block leaves the player nothing to chew on if the next
one is late, which is heard as chopping. :class:`BlockBuffer` hands out spans
long enough to absorb that jitter.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

_DONE = object()


def detach(source: Callable[[], Iterable[Any]]) -> Iterator[Any]:
    """Yield what ``source`` produces, consuming it on a separate thread.

    An exception raised by the source is forwarded to the caller rather than
    swallowed, so a failing turn still surfaces where it can be reported.
    """
    box: queue.Queue[Any] = queue.Queue()

    def run() -> None:
        try:
            for item in source():
                box.put(item)
        except Exception as exc:
            box.put(exc)
        finally:
            box.put(_DONE)

    threading.Thread(target=run, daemon=True).start()
    while True:
        item = box.get()
        if item is _DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


class BlockBuffer:
    """Group audio chunks into spans long enough to play without starving.

    The first span is longer than the rest: it has to cover the jitter of every
    later one, since a player that runs dry once is heard chopping.
    """

    def __init__(self, sample_rate: int, *, prebuffer_s: float = 2.0, block_s: float = 1.0) -> None:
        self._prebuffer = int(sample_rate * prebuffer_s)
        self._block = int(sample_rate * block_s)
        self._pending: list[np.ndarray] = []
        self._count = 0
        self._started = False

    @property
    def _target(self) -> int:
        return self._block if self._started else self._prebuffer

    def push(self, samples: np.ndarray) -> np.ndarray | None:
        """Take one chunk; return a span once enough has accumulated."""
        flat = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._pending.append(flat)
        self._count += flat.size
        if self._count < self._target:
            return None
        self._started = True
        return self.flush()

    def flush(self) -> np.ndarray | None:
        """Return whatever is held, so the tail of a reply is never dropped."""
        if not self._pending:
            return None
        span = np.concatenate(self._pending)
        self._pending = []
        self._count = 0
        return span
