"""``EndpointResponseGenerator`` — answers from a serverless endpoint.

The counterpart of :class:`~lfm2_audio.evaluation.model_generator.ModelResponseGenerator`
for a variant that is deployed rather than loaded. Same contract, same
trajectory shape, so a run against an endpoint and a run against a local
checkpoint are read side by side without an asterisk.

This is what makes ``max_parallel`` worth raising: endpoints are independent
workers and the time goes into HTTP, not into a GPU the variants would have had
to share.
"""

from __future__ import annotations

import logging

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.evaluation.question import Question
from lfm2_audio.evaluation.turn_trajectory import TurnTrajectoryBuilder
from lfm2_audio.remote.client import LiquidAudioClient
from lfm2_audio.scorer.sample import EvalSample

logger = logging.getLogger(__name__)


class EndpointResponseGenerator:
    """Adapts a :class:`LiquidAudioClient` to the ``ResponseGenerator`` contract."""

    def __init__(self, client: LiquidAudioClient, *, max_tokens: int | None = None) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def generate(self, question: Question) -> EvalSample:
        audio = Waveform.from_file(question.audio_path) if question.audio_path else None
        # No history is sent: every case must start from the same context, or
        # latencies and answers stop being comparable across cases.
        reply = self._client.invoke(
            text=None if audio is not None else question.text,
            audio=audio,
            max_tokens=self._max_tokens,
        )

        trajectory = TurnTrajectoryBuilder(
            prompt_text=question.text,
            spoken_prompt=audio is not None,
        ).build(reply)

        return EvalSample(
            sample_id=question.question_id,
            prompt_text=question.text,
            prompt_audio=audio,
            predicted_text=reply.raw_text or reply.text,
            predicted_audio=reply.audio,
            reference_text=question.reference_answer,
            expected_calls=question.expected_calls,
            trajectory=trajectory.as_list(),
            metadata={**question.metadata, **reply.metrics.as_dict()},
        )
