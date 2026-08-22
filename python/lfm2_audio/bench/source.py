"""``AnswerSource`` — where the bench gets its answers from.

The bench cares that something answers a question; it does not care whether a
model is loaded in this process or reachable over HTTPS. Keeping that behind a
protocol is what lets the rating UI run on a laptop with no GPU, against a
serverless endpoint, using the same code that drives a local checkpoint.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.reply import Reply


@runtime_checkable
class AnswerSource(Protocol):
    """Produces one spoken answer per question."""

    @property
    def label(self) -> str:
        """Identifies what produced the answer — recorded with every verdict."""

    @property
    def keeps_history(self) -> bool:
        """Whether consecutive turns share a conversation context."""

    def answer(
        self,
        *,
        text: str | None = None,
        audio: Waveform | None = None,
        max_tokens: int = 400,
    ) -> Reply: ...

    def reset(self) -> None:
        """Drop any conversational context. A no-op for stateless sources."""
