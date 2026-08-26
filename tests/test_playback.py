"""BlockBuffer et detach : ce qui garde la lecture continue sans brider le modèle."""

from __future__ import annotations

import numpy as np
import pytest

from lfm2_audio.orchestrator.playback import BlockBuffer, detach

FRAME = 1920  # une frame Mimi = 80 ms à 24 kHz


def frame(value: float = 0.1) -> np.ndarray:
    return np.full(FRAME, value, dtype=np.float32)


class TestBlockBuffer:
    def test_should_hold_chunks_when_below_prebuffer(self) -> None:
        buffer = BlockBuffer(24_000, prebuffer_s=2.0, block_s=1.0)

        spans = [buffer.push(frame()) for _ in range(24)]  # 24 x 80 ms = 1,92 s

        assert spans == [None] * 24

    def test_should_release_span_when_prebuffer_reached(self) -> None:
        buffer = BlockBuffer(24_000, prebuffer_s=2.0, block_s=1.0)

        spans = [buffer.push(frame()) for _ in range(25)]  # 2,0 s atteint

        assert all(s is None for s in spans[:24])
        assert spans[24] is not None
        assert spans[24].size == 25 * FRAME

    def test_should_use_shorter_blocks_after_the_first_span(self) -> None:
        buffer = BlockBuffer(24_000, prebuffer_s=2.0, block_s=1.0)
        for _ in range(25):
            buffer.push(frame())

        spans = [buffer.push(frame()) for _ in range(13)]  # 13 x 80 ms = 1,04 s

        assert all(s is None for s in spans[:12])
        assert spans[12] is not None
        assert spans[12].size == 13 * FRAME

    def test_should_return_tail_on_flush(self) -> None:
        buffer = BlockBuffer(24_000, prebuffer_s=2.0, block_s=1.0)
        buffer.push(frame())

        tail = buffer.flush()

        assert tail is not None
        assert tail.size == FRAME

    def test_should_return_none_when_flushing_empty(self) -> None:
        assert BlockBuffer(24_000).flush() is None

    def test_should_preserve_samples_in_order(self) -> None:
        buffer = BlockBuffer(24_000, prebuffer_s=0.16, block_s=0.08)
        buffer.push(frame(0.25))
        span = buffer.push(frame(0.75))

        assert span is not None
        assert span[0] == pytest.approx(0.25)
        assert span[-1] == pytest.approx(0.75)


class TestDetach:
    def test_should_yield_every_item_in_order(self) -> None:
        assert list(detach(lambda: iter([1, 2, 3]))) == [1, 2, 3]

    def test_should_forward_source_exception(self) -> None:
        def boom() -> list[int]:
            raise ValueError("source en panne")

        with pytest.raises(ValueError, match="source en panne"):
            list(detach(boom))

    def test_should_not_block_source_on_slow_consumer(self) -> None:
        """Le producteur doit finir sans attendre que le consommateur avance."""
        done: list[bool] = []

        def source() -> list[int]:
            items = list(range(50))
            done.append(True)
            return items

        stream = detach(source)
        first = next(stream)

        assert first == 0
        assert list(stream) == list(range(1, 50))
        assert done == [True]
