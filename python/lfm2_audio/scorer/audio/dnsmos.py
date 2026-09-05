"""Historical import path; the implementation lives in the evaluation toolkit (avet)."""

from __future__ import annotations

from avet.scorers.audio.dnsmos_scorer import (
    DNSMOS_SAMPLE_RATE,
    INPUT_LENGTH_S,
    MODEL_ENV_VAR,
    SUBSCORES,
    DnsmosScorer,
    calibrate_p835,
    tile_to_window,
)

__all__ = [
    "DNSMOS_SAMPLE_RATE",
    "INPUT_LENGTH_S",
    "MODEL_ENV_VAR",
    "SUBSCORES",
    "DnsmosScorer",
    "calibrate_p835",
    "tile_to_window",
]
