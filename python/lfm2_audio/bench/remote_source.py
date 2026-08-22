"""``RemoteEndpointSource`` — answers from a serverless endpoint over HTTPS.

Lets the rating UI run on a machine with no GPU: the worker holds the model and
scales to zero between turns, so a listening session costs seconds of compute
rather than an hour of rented card.
"""

from __future__ import annotations

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.reply import Reply
from lfm2_audio.remote.client import LiquidAudioClient


class RemoteEndpointSource:
    """Adapts :class:`LiquidAudioClient` to the bench's ``AnswerSource`` protocol."""

    def __init__(self, client: LiquidAudioClient, *, label: str) -> None:
        self._client = client
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    @property
    def keeps_history(self) -> bool:
        """Serverless workers are stateless: every call starts from scratch.

        The Talk tab is therefore single-turn against an endpoint. Judging a
        model's *voice* is unaffected; judging how it handles a conversation is
        not possible this way, and the UI says so rather than pretending.
        """
        return False

    def answer(
        self,
        *,
        text: str | None = None,
        audio: Waveform | None = None,
        max_tokens: int = 400,
    ) -> Reply:
        return self._client.invoke(text=text, audio=audio, max_tokens=max_tokens)

    def reset(self) -> None:
        """No-op: there is no server-side history to clear."""
