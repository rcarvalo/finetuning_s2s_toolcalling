"""Historical import path; the implementation lives in the evaluation toolkit (avet)."""

from __future__ import annotations

from avet.scorers.audio.nisqa_scorer import DIMENSIONS, MODEL_ENV_VAR, NISQA_SAMPLE_RATE, NisqaScorer

__all__ = ["DIMENSIONS", "MODEL_ENV_VAR", "NISQA_SAMPLE_RATE", "NisqaScorer"]
