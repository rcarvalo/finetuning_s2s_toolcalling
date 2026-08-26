"""Keep the listener from throttling the model.

Gradio blocks the generator on every ``yield`` until the transport accepts the
message — about 200 ms over a ``gradio.live`` tunnel. With the agent consumed
inline, every round trip stalls generation: a turn producing 11 s of speech
took 35 s. :func:`detach` puts the agent on its own thread so it runs at its
own pace whatever the listener does.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import Any

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
