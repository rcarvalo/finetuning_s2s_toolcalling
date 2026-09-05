"""Historical import path; the implementation lives in the evaluation toolkit (avet)."""

from __future__ import annotations

from avet.scorers.audio.wer_scorer import WerScorer
from avet.text.wer import character_error_rate, normalize_transcript, word_error_rate

__all__ = ["WerScorer", "character_error_rate", "normalize_transcript", "word_error_rate"]
