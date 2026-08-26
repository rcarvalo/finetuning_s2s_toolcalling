"""``LocalTranscriber`` — what the microphone actually captured, in text.

Why not ask the served model? Measured on the endpoint (2026-08-25): asked to
transcribe, LFM2.5-Audio answers the question roughly half the time instead of
repeating it. A transcript that is silently an *answer* is worse than none —
it would misreport what the user said. faster-whisper is deterministic about
its job, runs on CPU, and pulls no torch, which keeps the voice client light
enough for a laptop or a Reachy Mini.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lfm2_audio.ds.audio import INPUT_SAMPLE_RATE, Waveform

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"


class LocalTranscriber:
    """Whisper transcription of a captured utterance, loaded on first use."""

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE, *, language: str | None = "en") -> None:
        self._model_size = model_size
        self._language = language
        self._model: WhisperModel | None = None

    def transcribe(self, wave: Waveform) -> str:
        """Text of the utterance, or an empty string if ASR is unavailable."""
        model = self._load()
        if model is None:
            return ""
        segments, _ = model.transcribe(
            wave.resample(INPUT_SAMPLE_RATE).samples,
            language=self._language,
            beam_size=1,
        )
        return " ".join(segment.text for segment in segments).strip()

    def _load(self) -> WhisperModel | None:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                logger.warning("faster-whisper absent: no transcript (uv sync --extra voice)")
                return None
            logger.info("loading whisper-%s (first run downloads the weights)", self._model_size)
            # int8 on CPU: a laptop and a Reachy Mini both have one, and neither
            # has a spare GPU while the audio model is already using it.
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        return self._model
