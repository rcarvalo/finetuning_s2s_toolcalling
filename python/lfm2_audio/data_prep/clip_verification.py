"""Does a synthesised clip say its text? The rule every brick applies.

An independent ASR re-listens to the clip; the transcript is compared with
the text in spoken form (numbers, titles) at the word and character level.
A clip passes on either rate: a misheard proper noun breaks the word rate on
a short sentence, a truncated or invented clip fails both.
"""

from __future__ import annotations

from avet.text.spoken_form import spoken_form
from avet.text.wer import character_error_rate, word_error_rate


def verification_rates(text: str, heard: str, lang: str) -> tuple[float, float]:
    """``(wer, cer)`` between what the clip should say and what the re-listen heard."""
    reference, hypothesis = spoken_form(text, lang), spoken_form(heard, lang)
    return word_error_rate(reference, hypothesis), character_error_rate(reference, hypothesis)


def accepted(wer: float, cer: float, *, max_wer: float, max_cer: float) -> bool:
    """Fine at the word level, or close enough at the character level."""
    return wer <= max_wer or cer <= max_cer
