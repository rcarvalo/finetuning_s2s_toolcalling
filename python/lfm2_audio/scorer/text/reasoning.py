"""Historical import path; the implementation lives in the evaluation toolkit (avet)."""

from __future__ import annotations

from avet.scorers.judge.reasoning_scorer import ReasoningScorer

from lfm2_audio.core.prompt import spoken_part

__all__ = ["ReasoningScorer", "spoken_part"]
