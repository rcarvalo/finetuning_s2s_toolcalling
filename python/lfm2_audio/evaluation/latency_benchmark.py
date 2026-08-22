"""``LatencyBenchmark`` — campagne de mesure de latence sur un modèle chargé."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from lfm2_audio.evaluation.latency import DEFAULT_PROMPTS, LatencyReport, format_ms
from lfm2_audio.evaluation.latency_sample import LatencySample
from lfm2_audio.serving.model import LFM2Audio

logger = logging.getLogger(__name__)


class LatencyBenchmark:
    """Campagne de mesure sur un modèle chargé.

    L'historique est vidé entre les tours : chaque mesure part du même contexte,
    sinon le prompt s'allonge et le TTFA dérive d'un run à l'autre.
    """

    def __init__(
        self,
        model: LFM2Audio,
        *,
        prompts: Sequence[str] = DEFAULT_PROMPTS,
        max_tokens: int = 192,
    ) -> None:
        self._model = model
        self._prompts = tuple(prompts)
        self._max_tokens = max_tokens

    def warmup(self, rounds: int) -> None:
        """Tours de chauffe non mesurés (JIT Triton, autotuning CUDA graphs)."""
        for index in range(rounds):
            sample = self._measure(index)
            logger.info(
                "warmup %d/%d : ttfa=%s frames=%d",
                index + 1,
                rounds,
                format_ms(sample.ttfa_s),
                sample.audio_frames,
            )

    def run(self, rounds: int) -> LatencyReport:
        """``rounds`` tours mesurés, en tournant sur les prompts."""
        return LatencyReport(samples=tuple(self._measure(index) for index in range(rounds)))

    def _measure(self, index: int) -> LatencySample:
        prompt = self._prompts[index % len(self._prompts)]
        self._model.reset()
        return LatencySample.from_reply(prompt, self._model.reply(text=prompt, max_tokens=self._max_tokens))
