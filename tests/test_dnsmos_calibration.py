"""DNSMOS P.835 calibration — the polynomials must span the MOS scale.

Regression: a cubic approximation used to saturate BAK below 2.0, so clean
speech scored like noisy speech and every OVRL was meaningless (~1.6 for the
whole EN baseline). Any calibration that cannot reach the top of the scale is
broken by construction, whatever its coefficients.
"""

from __future__ import annotations

import pytest

from lfm2_audio.scorer.audio.dnsmos import calibrate_p835

# Raw ONNX outputs observed on reference signals (L4, sig_bak_ovr.onnx).
NOISE_RAW = (1.033, 1.010, 1.021)
CLEAN_SPEECH_RAW = (4.2, 4.5, 3.5)


def test_should_map_white_noise_near_the_bottom_of_the_scale() -> None:
    sig, bak, ovrl = calibrate_p835(*NOISE_RAW)

    assert all(0.5 <= score <= 1.6 for score in (sig, bak, ovrl))


def test_should_map_clean_speech_to_a_high_background_score() -> None:
    _, bak, _ = calibrate_p835(*CLEAN_SPEECH_RAW)

    assert bak > 4.0, "clean synthetic speech must not score like noisy speech"


def test_should_reach_the_top_of_the_mos_scale() -> None:
    """A calibration that cannot exceed ~2 is unusable, whatever the audio."""
    sig, bak, ovrl = calibrate_p835(4.6, 4.9, 4.5)

    assert sig > 3.5
    assert bak > 4.0
    assert ovrl > 3.6


@pytest.mark.parametrize("index", [0, 1, 2])
def test_should_increase_with_raw_quality(index: int) -> None:
    low = calibrate_p835(2.0, 2.0, 2.0)[index]
    high = calibrate_p835(4.0, 4.0, 4.0)[index]

    assert high > low
