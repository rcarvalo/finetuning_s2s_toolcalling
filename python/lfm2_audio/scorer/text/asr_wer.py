"""``AsrWerScorer`` — WER of the model's TEXT reply against a reference transcript.

The audio ``wer`` scorer transcribes the model's generated speech; it measures
the TTS path. On an ASR benchmark the model listens to a clip and answers with
the transcript as text, so the measurement is a plain text-vs-text WER — no
transcriber involved, which also makes this the D1 gate metric (FLEURS-fr)
runnable anywhere.

Same Levenshtein and the same normalisation as the audio scorer
(:func:`lfm2_audio.scorer.audio.wer.word_error_rate`), so the two rates are
comparable when both are quoted in a report.
"""

from __future__ import annotations

from typing import ClassVar

from lfm2_audio.core.prompt import spoken_part
from lfm2_audio.scorer.audio.wer import word_error_rate
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample


class AsrWerScorer(BaseScorer):
    """Word Error Rate between the text reply and the reference transcript."""

    name = "asr_wer"
    higher_is_better: ClassVar[bool] = False
    description: ClassVar[str] = "WER de la réponse texte contre la transcription de référence"

    def __init__(self, *, normalize: bool = True) -> None:
        self._normalize = normalize

    def supports(self, sample: EvalSample) -> bool:
        return bool(spoken_part(sample.predicted_text).strip()) and bool(sample.reference_text.strip())

    def skip_reason(self, sample: EvalSample) -> str:
        if not spoken_part(sample.predicted_text).strip():
            return "aucune réponse texte à comparer"
        return "aucune transcription de référence"

    def measure(self, sample: EvalSample) -> ScoreResult:
        hypothesis = spoken_part(sample.predicted_text)
        reference = sample.reference_text
        rate = word_error_rate(reference, hypothesis, normalize=self._normalize)
        return ScoreResult.ok(
            self.name,
            rate,
            higher_is_better=False,
            details={"reference": reference, "hypothesis": hypothesis},
        )
