"""Get the generated speech back out of Inspect logs, for the gate metrics.

The extraction now lives in the evaluation toolkit
(:class:`avet.bridge.log_audio_extractor.LogAudioExtractor`); this module
keeps the historical :class:`LoggedReply` shape and functions, with the LFM2
text cleaner bound so ``spoken_text`` drops the call markers.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from avet.bridge.log_audio_extractor import LogAudioExtractor

from lfm2_audio.avet_components.text_cleaner import Lfm2TextCleaner
from lfm2_audio.core.prompt import spoken_part


@dataclass(frozen=True, slots=True)
class LoggedReply:
    """One reply of a campaign: its text, its speech, and where it landed."""

    sample_id: str
    text: str
    wav_path: Path
    seconds: float
    question_language: str | None

    @property
    def spoken_text(self) -> str:
        """What the model said, markers stripped — what the audio should match."""
        return spoken_part(self.text)


def latest_log(log_dir: Path) -> Path:
    """The most recent ``.eval`` under ``log_dir``."""
    return LogAudioExtractor.latest_log(log_dir)


def extract_replies(log_path: Path, audio_out: Path, *, limit: int | None = None) -> Iterator[LoggedReply]:
    """Write each reply's speech to ``audio_out`` and yield what it was."""
    for reply in LogAudioExtractor(Lfm2TextCleaner()).extract(log_path, audio_out, limit=limit):
        yield LoggedReply(
            sample_id=reply.sample_id,
            text=reply.text,
            wav_path=reply.wav_path,
            seconds=reply.seconds,
            question_language=reply.question_language,
        )


__all__ = ["LoggedReply", "extract_replies", "latest_log"]
