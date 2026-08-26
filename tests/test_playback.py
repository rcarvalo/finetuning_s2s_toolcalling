"""detach : le consommateur ne doit jamais dicter sa vitesse au producteur."""

from __future__ import annotations

import pytest

from lfm2_audio.orchestrator.playback import detach


class TestDetach:
    def test_should_yield_every_item_in_order(self) -> None:
        assert list(detach(lambda: iter([1, 2, 3]))) == [1, 2, 3]

    def test_should_yield_nothing_for_an_empty_source(self) -> None:
        assert list(detach(lambda: iter([]))) == []

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
