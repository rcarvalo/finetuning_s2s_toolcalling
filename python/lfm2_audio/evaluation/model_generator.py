"""``ModelResponseGenerator`` — fait répondre un modèle LFM2.5-Audio chargé."""

from __future__ import annotations

import logging

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.evaluation.question import Question
from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.serving.model import LFM2Audio

logger = logging.getLogger(__name__)


class ModelResponseGenerator:
    """Adapte un :class:`LFM2Audio` au contrat ``ResponseGenerator``.

    L'historique est vidé entre deux questions : sans cela le contexte
    s'accumule, les réponses dérivent et les latences ne sont plus comparables
    d'un cas à l'autre.
    """

    def __init__(self, model: LFM2Audio, *, max_tokens: int | None = None) -> None:
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, question: Question) -> EvalSample:
        self._model.reset()

        audio = Waveform.from_file(question.audio_path) if question.audio_path else None
        reply = self._model.reply(
            text=None if audio is not None else question.text,
            audio=audio,
            max_tokens=self._max_tokens,
        )

        return EvalSample(
            sample_id=question.question_id,
            prompt_text=question.text,
            prompt_audio=audio,
            # raw_text : les marqueurs <|tool_call_*|> doivent survivre au scoring.
            predicted_text=reply.raw_text or reply.text,
            predicted_audio=reply.audio,
            reference_text=question.reference_answer,
            expected_calls=question.expected_calls,
            metadata={**question.metadata, **reply.metrics.as_dict()},
        )
