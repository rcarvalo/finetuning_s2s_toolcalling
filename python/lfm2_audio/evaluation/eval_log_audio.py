"""Get the generated speech back out of Inspect logs, for the gate metrics.

Every gate in the bilingual plan is decided on VERSA numbers computed over the
audio a campaign produced. That audio already exists inside the ``.eval`` log
(base64, next to the reply it belongs to), so the gate pass must never
re-generate it: re-running the model would measure a different sample of its
own randomness than the campaign the gate is about.

This module is the extraction half — pure IO over logs — and leaves the scoring
to :class:`~lfm2_audio.evaluation.versa_runner.VersaRunner`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentAudio

from lfm2_audio.core.prompt import spoken_part
from lfm2_audio.inspect_bridge.audio import data_uri_to_waveform

logger = logging.getLogger(__name__)


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
    """The most recent ``.eval`` under ``log_dir``.

    Campaigns are re-run after fixes, and several attempts share a directory;
    reading the newest is what "the results of this campaign" means.
    """
    logs = sorted(log_dir.glob("**/*.eval"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"aucun .eval sous {log_dir}")
    return logs[-1]


def extract_replies(log_path: Path, audio_out: Path, *, limit: int | None = None) -> Iterator[LoggedReply]:
    """Write each reply's speech to ``audio_out`` and yield what it was."""
    audio_out.mkdir(parents=True, exist_ok=True)
    log = read_eval_log(str(log_path))
    written = 0
    for sample in log.samples or []:
        if limit is not None and written >= limit:
            return
        message = sample.output.message if sample.output else None
        if message is None or isinstance(message.content, str):
            continue
        parts = [part for part in message.content if isinstance(part, ContentAudio)]
        if not parts:
            continue
        waveform = data_uri_to_waveform(parts[0].audio)
        path = audio_out / f"{sample.id}.wav"
        sf.write(str(path), waveform.samples, waveform.sample_rate, subtype="PCM_16")
        written += 1
        yield LoggedReply(
            sample_id=str(sample.id),
            text=sample.output.completion if sample.output else "",
            wav_path=path,
            seconds=len(waveform.samples) / waveform.sample_rate,
            question_language=(sample.metadata or {}).get("lang"),
        )
    logger.info("%d réponses audio extraites de %s", written, log_path.name)
