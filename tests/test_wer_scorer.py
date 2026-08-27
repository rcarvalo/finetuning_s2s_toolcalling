"""Tests du WER et de son scorer (transcripteur factice, sans GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from lfm2_audio.ds.audio import OUTPUT_SAMPLE_RATE, Waveform
from lfm2_audio.scorer.audio.wer import WerScorer, normalize_transcript, word_error_rate
from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.status import ScoreStatus


class FakeTranscriber:
    """Rend une transcription fixée. Satisfait le protocole ``Transcriber``."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0
        self.languages: list[str | None] = []

    def transcribe(self, audio: Waveform, *, language: str | None = None) -> str:
        self.calls += 1
        self.languages.append(language)
        return self._text


def _audio() -> Waveform:
    return Waveform.of(np.zeros(OUTPUT_SAMPLE_RATE, dtype=np.float32), OUTPUT_SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# Métrique
# --------------------------------------------------------------------------- #


def test_identical_texts_should_score_zero():
    assert word_error_rate("the weather in tokyo", "the weather in tokyo") == 0.0


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("a b c d", "a b c", 0.25),  # une suppression
        ("a b c d", "a b c d e", 0.25),  # une insertion
        ("a b c d", "a b x d", 0.25),  # une substitution
        ("a b", "x y", 1.0),  # tout faux
    ],
)
def test_should_count_each_edit_operation(reference, hypothesis, expected):
    assert word_error_rate(reference, hypothesis) == pytest.approx(expected)


def test_empty_reference_should_score_zero_against_empty_hypothesis():
    assert word_error_rate("", "") == 0.0


def test_empty_reference_should_score_one_against_content():
    assert word_error_rate("", "something") == 1.0


def test_normalisation_should_neutralise_case_and_punctuation():
    # L'ASR invente de la ponctuation : la pénaliser ne mesure pas l'intelligibilité.
    assert word_error_rate("The weather, in Tokyo!", "the weather in tokyo") == 0.0


def test_without_normalisation_punctuation_counts():
    assert word_error_rate("the weather!", "the weather", normalize=False) > 0.0


def test_normalize_transcript_should_collapse_whitespace():
    assert normalize_transcript("  Hello,   WORLD!  ") == "hello world"


# --------------------------------------------------------------------------- #
# Scorer
# --------------------------------------------------------------------------- #


def test_should_compare_transcription_to_the_reference():
    scorer = WerScorer(FakeTranscriber("the weather in tokyo"))
    sample = EvalSample(sample_id="s1", predicted_audio=_audio(), reference_text="the weather in tokyo")

    result = scorer.score(sample)

    assert result.value == 0.0
    assert result.higher_is_better is False
    assert result.details["hypothesis"] == "the weather in tokyo"


def test_should_fall_back_to_the_generated_text_as_reference():
    """En S2S interleaved, le WER mesure la fidélité du TTS à sa propre sortie texte."""
    scorer = WerScorer(FakeTranscriber("hello there"))
    sample = EvalSample(sample_id="s1", predicted_audio=_audio(), predicted_text="hello there<|text_end|>")

    assert scorer.score(sample).value == 0.0


def test_should_skip_a_sample_without_audio():
    scorer = WerScorer(FakeTranscriber("whatever"))

    result = scorer.score(EvalSample(sample_id="s1", reference_text="hi"))

    assert result.status is ScoreStatus.SKIPPED
    assert "audio" in result.reason


def test_should_skip_a_sample_without_reference():
    scorer = WerScorer(FakeTranscriber("whatever"))

    result = scorer.score(EvalSample(sample_id="s1", predicted_audio=_audio()))

    assert result.status is ScoreStatus.SKIPPED


def test_should_not_transcribe_when_skipping():
    transcriber = FakeTranscriber("whatever")
    WerScorer(transcriber).score(EvalSample(sample_id="s1"))

    assert transcriber.calls == 0


def test_should_transcribe_in_the_sample_language() -> None:
    """A bilingual campaign must grade each sample in ITS language: forcing EN
    on FR speech measures the ASR's confusion, not intelligibility."""
    transcriber = FakeTranscriber("bonjour tout le monde")
    scorer = WerScorer(transcriber)
    sample = EvalSample(
        sample_id="fr1",
        predicted_text="bonjour tout le monde",
        predicted_audio=_audio(),
        metadata={"lang": "fr"},
    )

    scorer.score(sample)

    assert transcriber.languages == ["fr"]


def test_should_fall_back_to_the_transcriber_default_without_lang() -> None:
    """No language in the metadata and none detectable in the reply: the
    transcriber's own default is all that is left."""
    transcriber = FakeTranscriber("1969")
    scorer = WerScorer(transcriber)
    sample = EvalSample(sample_id="en1", predicted_text="1969", predicted_audio=_audio())

    scorer.score(sample)

    assert transcriber.languages == [None]


def test_should_transcribe_in_the_language_the_model_actually_spoke() -> None:
    """A model that does not mirror answers a FR question in EN. Forcing FR ASR
    on that reply measures the ASR's confusion: on the 0B baseline it inflated
    the roundtrip WER from 0.53 to 0.86."""
    transcriber = FakeTranscriber("the weather is sunny today")
    scorer = WerScorer(transcriber)
    sample = EvalSample(
        sample_id="s1",
        predicted_text="The weather is sunny today and it will stay warm.",
        predicted_audio=_audio(),
        metadata={"lang": "fr"},  # la QUESTION est en français
    )

    scorer.score(sample)

    assert transcriber.languages == ["en"]


def test_question_language_should_remain_the_fallback() -> None:
    """Too short to classify: the question's language is the best guess left."""
    transcriber = FakeTranscriber("42")
    scorer = WerScorer(transcriber)
    sample = EvalSample(sample_id="s1", predicted_text="42", predicted_audio=_audio(), metadata={"lang": "fr"})

    scorer.score(sample)

    assert transcriber.languages == ["fr"]
