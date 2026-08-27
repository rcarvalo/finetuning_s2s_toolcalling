"""``WerScorer`` — intelligibilité de l'audio généré.

On transcrit la sortie du modèle et on compare à ce qu'il était censé dire. Un
WER qui grimpe après un fine-tuning est le signal le plus précoce que les têtes
audio ont dérivé — bien avant qu'un MOS ne bouge.

Le transcripteur est **injecté** : ce scorer dépend du fait de transcrire, pas
de Whisper. Il reste donc testable sans GPU.
"""

from __future__ import annotations

import re
from typing import ClassVar

from lfm2_audio.scorer.audio.transcriber import Transcriber
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.text.lang_match import detect_language

_PUNCTUATION = re.compile(r"[^\w\s']", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


class WerScorer(BaseScorer):
    """Word Error Rate entre la transcription de l'audio généré et sa référence."""

    name = "wer"
    higher_is_better: ClassVar[bool] = False
    description: ClassVar[str] = "taux d'erreur mot de l'audio généré, re-transcrit"

    def __init__(self, transcriber: Transcriber, *, normalize: bool = True) -> None:
        self._transcriber = transcriber
        self._normalize = normalize

    def supports(self, sample: EvalSample) -> bool:
        return sample.has_predicted_audio and bool(sample.spoken_reference.strip())

    def skip_reason(self, sample: EvalSample) -> str:
        if not sample.has_predicted_audio:
            return "aucun audio généré à transcrire"
        return "aucune référence à comparer"

    def measure(self, sample: EvalSample) -> ScoreResult:
        audio = sample.predicted_audio
        if audio is None:  # supports() l'a déjà vérifié — ceinture et bretelles
            return ScoreResult.skipped(self.name, "aucun audio généré à transcrire")
        language = self._language_of(sample)
        hypothesis = self._transcriber.transcribe(audio, language=language)
        reference = sample.spoken_reference

        rate = word_error_rate(reference, hypothesis, normalize=self._normalize)
        return ScoreResult.ok(
            self.name,
            rate,
            higher_is_better=False,
            details={"reference": reference, "hypothesis": hypothesis, "asr_language": language},
        )

    @staticmethod
    def _language_of(sample: EvalSample) -> str | None:
        """The language to transcribe in: the one the model actually SPOKE.

        Not the question's ``metadata["lang"]``. A model that does not yet
        mirror answers a French question in English, and forcing French ASR on
        that reply measures the ASR's confusion, not the speech: on the 0B
        baseline it inflated the roundtrip WER from 0.53 to 0.86. The reply
        text is what produced the audio, so it is what says which ASR to use;
        the question's language remains the fallback when the text carries no
        clear signal.
        """
        return detect_language(sample.predicted_text) or sample.metadata.get("lang")


def normalize_transcript(text: str) -> str:
    """Minuscules, ponctuation retirée, espaces normalisés.

    Sans cette normalisation le WER pénalise la ponctuation inventée par l'ASR,
    ce qui n'a rien à voir avec l'intelligibilité qu'on cherche à mesurer.
    """
    lowered = text.lower().strip()
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", lowered)).strip()


def word_error_rate(reference: str, hypothesis: str, *, normalize: bool = True) -> float:
    """WER = (substitutions + insertions + suppressions) / mots de référence.

    Distance de Levenshtein au niveau des mots, en O(len(ref) × len(hyp)) mémoire
    linéaire. Une référence vide rend 0.0 si l'hypothèse l'est aussi, 1.0 sinon.
    """
    ref_words = (normalize_transcript(reference) if normalize else reference).split()
    hyp_words = (normalize_transcript(hypothesis) if normalize else hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    previous = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp_words, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[j] + 1,  # suppression
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current

    return previous[-1] / len(ref_words)
