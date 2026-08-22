"""Tests du value object ``ScoreResult`` et de son statut."""

from __future__ import annotations

import pytest

from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.status import ScoreStatus


def test_ok_should_carry_a_value():
    result = ScoreResult.ok("wer", 0.12, higher_is_better=False)

    assert result.status is ScoreStatus.OK
    assert result.value == 0.12
    assert result.higher_is_better is False
    assert result.is_measurement


@pytest.mark.parametrize(
    ("factory", "status"),
    [
        (ScoreResult.unavailable, ScoreStatus.UNAVAILABLE),
        (ScoreResult.skipped, ScoreStatus.SKIPPED),
        (ScoreResult.failed, ScoreStatus.FAILED),
    ],
)
def test_failure_constructors_should_never_carry_a_value(factory, status):
    result = factory("dnsmos", "poids manquants")

    assert result.status is status
    assert result.value is None
    assert not result.is_measurement
    assert result.reason == "poids manquants"


def test_only_ok_counts_as_a_measurement():
    # Distinguer « non mesuré » de « mesuré à zéro » est tout l'intérêt du statut.
    assert ScoreStatus.OK.is_measurement
    assert not ScoreStatus.UNAVAILABLE.is_measurement
    assert not ScoreStatus.SKIPPED.is_measurement
    assert not ScoreStatus.FAILED.is_measurement


def test_as_dict_should_omit_empty_fields():
    payload = ScoreResult.ok("tool_call", 1.0).as_dict()

    assert payload == {"scorer": "tool_call", "status": "ok", "value": 1.0}


def test_as_dict_should_keep_reason_and_details():
    payload = ScoreResult.unavailable("nisqa", "checkpoint absent").as_dict()

    assert payload["reason"] == "checkpoint absent"
    assert "value" not in payload
