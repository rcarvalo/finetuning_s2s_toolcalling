"""``LangMatchScorer`` — does the reply come back in the user's language?

The mirroring gate (R3: ≥95 % on lang_mirror) needs a deterministic, free
measurement — a MOS-style judge costs an API call per sample and cannot be
run inside the training loop. For the FR/EN pair, function words separate the
two languages reliably at reply length; this is a two-class heuristic, not a
language identifier, and says so in its name.

Score is binary: 1.0 when the detected language matches ``expected_lang``
(falling back to ``lang``) in the sample metadata, 0.0 otherwise.
"""

from __future__ import annotations

import re
from typing import ClassVar

from lfm2_audio.core.prompt import spoken_part
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample

_WORD = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœ']+")

# High-frequency function words with no overlap between the two lists.
_FR_MARKERS = frozenset(
    [
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "et",
        "est",
        "sont",
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "elles",
        "ne",
        "pas",
        "que",
        "qui",
        "dans",
        "pour",
        "avec",
        "sur",
        "mais",
        "ou",
        "où",
        "donc",
        "car",
        "ce",
        "cette",
        "ces",
        "son",
        "sa",
        "ses",
        "mon",
        "ma",
        "mes",
        "ton",
        "ta",
        "tes",
        "votre",
        "vos",
        "notre",
        "nos",
        "leur",
        "leurs",
        "été",
        "être",
        "avoir",
        "fait",
        "faire",
        "plus",
        "très",
        "bien",
        "aussi",
        "comme",
        "tout",
        "tous",
        "toute",
        "toutes",
        "quel",
        "quelle",
        "oui",
        "non",
        "merci",
        "bonjour",
        "voilà",
        "ça",
        "c'est",
        "d'accord",
        "aujourd'hui",
    ]
)
_EN_MARKERS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "is",
        "are",
        "was",
        "were",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "it",
        "not",
        "that",
        "which",
        "in",
        "for",
        "with",
        "on",
        "but",
        "or",
        "so",
        "because",
        "this",
        "these",
        "those",
        "his",
        "her",
        "its",
        "my",
        "your",
        "our",
        "their",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "more",
        "very",
        "well",
        "also",
        "as",
        "all",
        "what",
        "yes",
        "no",
        "thanks",
        "hello",
        "there",
        "it's",
        "don't",
        "i'm",
        "you're",
        "today",
        "would",
        "could",
        "should",
    ]
)


def detect_language(text: str) -> str | None:
    """``"fr"``, ``"en"``, or ``None`` when the text carries no clear signal."""
    words = _WORD.findall(text.lower())
    fr = sum(1 for w in words if w in _FR_MARKERS)
    en = sum(1 for w in words if w in _EN_MARKERS)
    if fr == en:
        return None
    return "fr" if fr > en else "en"


class LangMatchScorer(BaseScorer):
    """Binary: the reply's detected language matches the expected one."""

    name = "lang_match"
    higher_is_better: ClassVar[bool] = True
    description: ClassVar[str] = "la réponse est-elle dans la langue attendue ? (FR/EN, déterministe)"

    def supports(self, sample: EvalSample) -> bool:
        return bool(self._expected(sample)) and bool(spoken_part(sample.predicted_text).strip())

    def skip_reason(self, sample: EvalSample) -> str:
        if not self._expected(sample):
            return "pas de langue attendue dans les métadonnées (expected_lang/lang)"
        return "aucune réponse texte à classifier"

    def measure(self, sample: EvalSample) -> ScoreResult:
        expected = self._expected(sample)
        reply = spoken_part(sample.predicted_text)
        detected = detect_language(reply)
        if detected is None:
            return ScoreResult.failed(self.name, f"langue indétectable sur : {reply[:80]!r}")
        return ScoreResult.ok(
            self.name,
            1.0 if detected == expected else 0.0,
            details={"expected": expected, "detected": detected},
        )

    @staticmethod
    def _expected(sample: EvalSample) -> str | None:
        expected = sample.metadata.get("expected_lang") or sample.metadata.get("lang")
        return str(expected) if expected in ("fr", "en") else None
