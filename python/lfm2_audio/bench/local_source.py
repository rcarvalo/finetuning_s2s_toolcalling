"""``LocalModelSource`` — answers from a model loaded in this process."""

from __future__ import annotations

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.reply import Reply
from lfm2_audio.serving.model import LFM2Audio


class LocalModelSource:
    """Adapts :class:`LFM2Audio` to the bench's ``AnswerSource`` protocol.

    Requires a GPU. Keeps conversation history, so the Talk tab is genuinely
    multi-turn here — unlike a stateless remote endpoint.
    """

    def __init__(self, model: LFM2Audio, *, label: str | None = None) -> None:
        self._model = model
        self._label = label or f"{model.checkpoint.name}@{model.backend_name}"

    @property
    def label(self) -> str:
        return self._label

    @property
    def keeps_history(self) -> bool:
        return True

    def answer(
        self,
        *,
        text: str | None = None,
        audio: Waveform | None = None,
        max_tokens: int = 400,
    ) -> Reply:
        return self._model.reply(text=text, audio=audio, max_tokens=max_tokens)

    def reset(self) -> None:
        self._model.reset()
