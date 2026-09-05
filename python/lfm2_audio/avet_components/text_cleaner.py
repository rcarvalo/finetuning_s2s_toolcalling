"""``Lfm2TextCleaner`` — what the model actually said, LFM2 markers and tool-call spans removed."""

from __future__ import annotations

from avet.text.text_cleaner import TEXT_CLEANERS

from lfm2_audio.core.prompt import spoken_part


@TEXT_CLEANERS.register("lfm2")
class Lfm2TextCleaner:
    """The whole ``<|tool_call_start|>…<|tool_call_end|>`` span goes, then every ``<|…|>`` token."""

    def clean(self, text: str) -> str:
        return spoken_part(text)
