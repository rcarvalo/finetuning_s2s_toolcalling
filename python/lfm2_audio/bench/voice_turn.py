"""``VoiceTurnHandler`` — one spoken utterance in, streamed reply audio out."""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
import numpy.typing as npt

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.remote.client import LiquidAudioClient

logger = logging.getLogger(__name__)

# Below this RMS the "utterance" is a VAD false trigger (silence, echo tail,
# keyboard noise): sending it would cost a full endpoint round-trip and the
# model would answer to nothing.
_MIN_RMS = 5e-4


class VoiceTurnHandler:
    """Bridge between a push-to-talk-free UI and the serverless endpoint.

    Deliberately fastrtc-agnostic: consumes and yields plain ``(sample_rate,
    samples)`` tuples so the turn logic is testable without WebRTC. The UI's
    pause detector calls :meth:`respond` once per utterance; chunks are
    yielded as they leave the endpoint, so playback starts at the first chunk
    (~TTFA + one poll) instead of after the full answer — this is what the
    aggregate-mode bench app cannot do.

    When ``save_dir`` is set, each turn's outgoing capture and incoming reply
    are written as ``<stamp>_user.wav`` / ``<stamp>_reply.wav``. This is the
    debugging tool for "the model doesn't understand me": listen to the user
    WAV to hear exactly what the model heard (degraded Bluetooth capture,
    clipped VAD window, wrong rate — the ear finds it in seconds).

    The v1 endpoint is stateless: every utterance is its own single-turn
    conversation. Session memory will come with the handler's session support.
    """

    def __init__(
        self,
        client: LiquidAudioClient,
        *,
        max_tokens: int | None = None,
        save_dir: Path | None = None,
    ) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._save_dir = save_dir

    def respond(
        self, audio: tuple[int, npt.NDArray[np.int16]]
    ) -> Generator[tuple[int, npt.NDArray[np.float32]], None, None]:
        """One turn: mic PCM in, reply chunks out as they are generated.

        The capture is sent at its native rate: the WAV payload carries the
        sample rate and the worker resamples for the encoder on the GPU
        (``vllm_omni.py``). Resampling here would drag torch into an app
        meant to run on a torch-less client (laptop, Reachy Mini).
        """
        sample_rate, samples = audio
        question = Waveform.from_pcm16(np.asarray(samples), int(sample_rate))
        if question.rms < _MIN_RMS:
            logger.info("utterance discarded (rms=%.5f): silence or echo", question.rms)
            return
        stamp = time.strftime("%H%M%S")
        self._save(question, f"{stamp}_user")
        logger.info(
            "utterance of %.1fs at %d Hz (rms=%.4f) sent to the endpoint",
            question.duration_s,
            question.sample_rate,
            question.rms,
        )
        for chunk in self._client.invoke_stream(audio=question, max_tokens=self._max_tokens):
            yield chunk.sample_rate, chunk.samples
        self._log_reply(stamp)

    def _save(self, wave: Waveform, name: str) -> None:
        if self._save_dir is None:
            return
        path = wave.save(self._save_dir / f"{name}.wav")
        logger.info("saved %s", path)

    def _log_reply(self, stamp: str) -> None:
        reply = self._client.last_reply
        if reply is None:
            return
        if reply.audio is not None:
            self._save(reply.audio, f"{stamp}_reply")
        metrics = reply.metrics
        logger.info(
            "reply (ttfa=%s total=%.1fs): %s",
            f"{metrics.ttfa_s:.2f}s" if metrics.ttfa_s is not None else "?",
            metrics.total_s,
            reply.text,
        )
