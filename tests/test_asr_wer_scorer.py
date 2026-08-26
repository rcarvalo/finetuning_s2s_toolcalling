"""Tests of the text-vs-text WER scorer (the D1 ASR gate metric)."""

from __future__ import annotations

import pytest

from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.status import ScoreStatus
from lfm2_audio.scorer.text.asr_wer import AsrWerScorer


def test_should_score_zero_on_an_exact_transcript() -> None:
    scorer = AsrWerScorer()
    sample = EvalSample(
        sample_id="a1",
        predicted_text="bonjour à tous",
        reference_text="bonjour à tous",
    )

    result = scorer.score(sample)

    assert result.value == 0.0
    assert result.higher_is_better is False


def test_should_count_word_errors_against_the_reference() -> None:
    scorer = AsrWerScorer()
    sample = EvalSample(
        sample_id="a1",
        predicted_text="bonjour à vous",
        reference_text="bonjour à tous",
    )

    result = scorer.score(sample)

    assert result.value == pytest.approx(1 / 3)


def test_should_ignore_speech_markers_in_the_reply() -> None:
    """The model interleaves text markers; they are not words it got wrong."""
    scorer = AsrWerScorer()
    sample = EvalSample(
        sample_id="a1",
        predicted_text="bonjour à tous<|text_end|>",
        reference_text="bonjour à tous",
    )

    assert scorer.score(sample).value == 0.0


def test_should_skip_without_a_reference() -> None:
    scorer = AsrWerScorer()

    result = scorer.score(EvalSample(sample_id="a1", predicted_text="bonjour"))

    assert result.status is ScoreStatus.SKIPPED
    assert "référence" in result.reason


def test_should_skip_without_a_text_reply() -> None:
    scorer = AsrWerScorer()

    result = scorer.score(EvalSample(sample_id="a1", reference_text="bonjour"))

    assert result.status is ScoreStatus.SKIPPED
