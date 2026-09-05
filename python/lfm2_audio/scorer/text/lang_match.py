"""Historical import path; the implementation lives in the evaluation toolkit (avet)."""

from __future__ import annotations

from avet.scorers.text.lang_match_scorer import LangMatchScorer
from avet.text.language import detect_language

__all__ = ["LangMatchScorer", "detect_language"]
