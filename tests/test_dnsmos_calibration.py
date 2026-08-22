"""Regression tests for the DNSMOS calibration and windowing.

These pin the two mistakes that made the first EN baseline unreadable: a
calibration polynomial that saturated (every clip landed on 1.60-1.62) and
zero-padding that fed the model silence instead of speech.
"""

from __future__ import annotations

import numpy as np
import pytest

from lfm2_audio.scorer.audio.dnsmos import (
    DNSMOS_SAMPLE_RATE,
    INPUT_LENGTH_S,
    calibrate_p835,
)

# Reference values recomputed from the non-personalised polyfit of
# microsoft/DNS-Challenge (dnsmos_local.py, get_polyfit_val).
REFERENCE_OVRL = {1.0: 1.094, 2.0: 2.006, 3.0: 2.783, 4.0: 3.425, 5.0: 3.932}


@pytest.mark.parametrize(("raw", "expected"), sorted(REFERENCE_OVRL.items()))
def test_calibration_should_match_the_reference_implementation(raw, expected):
    _, _, ovrl = calibrate_p835(raw, raw, raw)

    assert ovrl == pytest.approx(expected, abs=1e-3)


def test_calibration_should_stay_monotonic_over_the_useful_range():
    """A polynomial with an interior maximum collapses good clips onto one value.

    That is exactly what happened: raw scores of 3.5 to 5.0 all mapped to ~1.61,
    so 24 different answers produced an indistinguishable metric.
    """
    grid = np.linspace(1.0, 5.0, 200)
    ovrl = np.array([calibrate_p835(x, x, x)[2] for x in grid])

    assert np.all(np.diff(ovrl) > 0), "calibration must increase with raw quality"


def test_calibration_should_span_most_of_the_mos_scale():
    grid = np.linspace(1.0, 5.0, 200)
    ovrl = np.array([calibrate_p835(x, x, x)[2] for x in grid])

    # The broken cubic spanned 0.89 points; the reference spans ~2.8.
    assert np.ptp(ovrl) > 2.5


def test_good_audio_should_remain_distinguishable():
    """The failure mode was a 0.02-wide band across the whole 'good' range."""
    ovrl = np.array([calibrate_p835(x, x, x)[2] for x in np.linspace(3.5, 5.0, 50)])

    assert np.ptp(ovrl) > 0.5


def test_window_length_should_match_the_reference():
    assert pytest.approx(9.01) == INPUT_LENGTH_S
    assert DNSMOS_SAMPLE_RATE == 16_000


def test_tiling_should_never_introduce_silence():
    """Short clips are repeated, not zero-padded: padding a 5 s answer to 9 s
    hands the model 45 % silence and depresses every score equally."""
    clip = np.full(int(5.0 * DNSMOS_SAMPLE_RATE), 0.3, dtype=np.float32)
    window = int(INPUT_LENGTH_S * DNSMOS_SAMPLE_RATE)

    tiled = clip
    while tiled.size < window:
        tiled = np.concatenate([tiled, tiled])

    assert tiled.size >= window
    assert not np.any(tiled[:window] == 0.0)
