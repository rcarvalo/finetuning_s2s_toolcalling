"""``SpendMeter`` — le plafond qui protège un budget de 10 €.

Le compteur refuse l'appel SUIVANT dès que le plafond est atteint : le
dépassement possible est borné par une seule réponse, jamais par un run.
"""

from __future__ import annotations

import pytest

from lfm2_audio.scorer.text.llm_spend import (
    BATCH_DISCOUNT,
    SpendCapReachedError,
    SpendMeter,
    UnknownModelPriceError,
    price_of,
)


class TestSpendMeter:
    def test_should_start_at_zero(self) -> None:
        meter = SpendMeter("claude-sonnet-5")

        assert meter.usd == 0.0
        assert meter.calls == 0

    def test_should_price_input_and_output_at_their_own_rates(self) -> None:
        meter = SpendMeter("claude-sonnet-5")  # 2 $ / 10 $ par million

        meter.add(input_tokens=1_000_000, output_tokens=100_000)

        assert meter.usd == pytest.approx(2.0 + 1.0)

    def test_should_apply_the_batch_discount(self) -> None:
        meter = SpendMeter("claude-sonnet-5", discount=BATCH_DISCOUNT)

        meter.add(input_tokens=0, output_tokens=1_000_000)

        assert meter.usd == pytest.approx(5.0)

    def test_should_let_calls_through_under_the_cap(self) -> None:
        meter = SpendMeter("claude-sonnet-5", max_usd=1.0)
        meter.add(0, 50_000)  # 0,5 $

        meter.check()

    def test_should_refuse_the_next_call_once_the_cap_is_reached(self) -> None:
        meter = SpendMeter("claude-sonnet-5", max_usd=1.0)
        meter.add(0, 100_000)  # exactement 1 $

        with pytest.raises(SpendCapReachedError, match=r"plafond 1\.00"):
            meter.check()

    def test_should_never_cap_without_a_cap(self) -> None:
        meter = SpendMeter("claude-opus-5")
        meter.add(10_000_000, 10_000_000)

        meter.check()

    def test_should_summarise_on_one_marked_line(self) -> None:
        meter = SpendMeter("claude-sonnet-5")
        meter.add(1_000, 2_000)

        line = meter.summary()

        assert line.startswith("===SPEND=== model=claude-sonnet-5 calls=1 in=1000 out=2000 usd=")

    def test_should_refuse_a_model_without_a_known_price(self) -> None:
        with pytest.raises(UnknownModelPriceError, match="claude-imaginaire"):
            SpendMeter("claude-imaginaire")


def test_price_of_should_name_the_known_models_when_asked_an_unknown_one() -> None:
    with pytest.raises(UnknownModelPriceError, match="claude-opus-5"):
        price_of("gpt-9")
