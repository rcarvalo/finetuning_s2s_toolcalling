"""``ScoringConfig`` — the toolkit's scoring config with this model family's defaults.

The toolkit is model-agnostic; what LFM2 adds is how its replies are cleaned
(``lfm2`` text cleaner) and how its tool calls are read (``lfm2`` parser).
Both are registered by :mod:`lfm2_audio.avet_components`.
"""

from __future__ import annotations

from avet.scoring.scorer_config import ScorerConfig
from avet.scoring.scoring_config import AsrBackend
from avet.scoring.scoring_config import ScoringConfig as _ScoringConfig


class ScoringConfig(_ScoringConfig):
    """LFM2 defaults on top of the toolkit's config."""

    text_cleaner: str = "lfm2"
    tool_call_parser: str = "lfm2"
    asr_backend: AsrBackend = "transformers"
    judge_model: str = "google/gemini-3.6-flash"
    """Judge through Inspect's model layer; ``GEMINI_API_KEY`` is bridged to ``GOOGLE_API_KEY``."""

    @classmethod
    def with_defaults(cls) -> ScoringConfig:
        """The historical default set: audio (wer, dnsmos, utmos) + text (tool_call, reasoning)."""
        names = ("wer", "dnsmos", "utmos", "tool_call", "reasoning")
        return cls(scorers=tuple(ScorerConfig(name=name) for name in names))


__all__ = ["ScorerConfig", "ScoringConfig"]
