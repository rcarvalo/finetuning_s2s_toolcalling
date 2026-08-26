"""Tests of the deterministic FR/EN language-mirroring scorer."""

from __future__ import annotations

import pytest

from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.status import ScoreStatus
from lfm2_audio.scorer.text.lang_match import LangMatchScorer, detect_language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bonjour, je suis ravi de vous aider avec cette question.", "fr"),
        ("Sure, the weather in Paris is sunny today and quite warm.", "en"),
        ("Il fait très beau aujourd'hui, c'est une belle journée pour sortir.", "fr"),
        ("I think you should take the train because it is faster.", "en"),
    ],
)
def test_detect_language_on_clear_sentences(text: str, expected: str) -> None:
    assert detect_language(text) == expected


def test_detect_language_should_abstain_without_signal() -> None:
    assert detect_language("42") is None


def test_should_score_one_when_the_reply_mirrors_the_language() -> None:
    scorer = LangMatchScorer()
    sample = EvalSample(
        sample_id="m1",
        predicted_text="Bien sûr, je peux vous expliquer cela tout de suite.",
        metadata={"lang": "fr", "expected_lang": "fr"},
    )

    result = scorer.score(sample)

    assert result.value == 1.0
    assert result.details["detected"] == "fr"


def test_should_score_zero_when_the_reply_switches_language() -> None:
    """The failure mode the benchmark exists for: FR question, EN answer."""
    scorer = LangMatchScorer()
    sample = EvalSample(
        sample_id="m2",
        predicted_text="Sure, here is what you should know about the weather.",
        metadata={"lang": "fr", "expected_lang": "fr"},
    )

    assert scorer.score(sample).value == 0.0


def test_expected_lang_should_win_over_lang() -> None:
    """Code-switch cases: the clip is tagged with the expected REPLY language."""
    scorer = LangMatchScorer()
    sample = EvalSample(
        sample_id="m3",
        predicted_text="Of course, the tallest building is in Dubai these days.",
        metadata={"lang": "fr", "expected_lang": "en"},
    )

    assert scorer.score(sample).value == 1.0


def test_should_skip_without_an_expected_language() -> None:
    scorer = LangMatchScorer()

    result = scorer.score(EvalSample(sample_id="m4", predicted_text="Bonjour tout le monde, je vais bien."))

    assert result.status is ScoreStatus.SKIPPED


def test_should_fail_rather_than_guess_on_an_unclassifiable_reply() -> None:
    scorer = LangMatchScorer()
    sample = EvalSample(sample_id="m5", predicted_text="1969.", metadata={"lang": "en"})

    result = scorer.score(sample)

    assert result.status is ScoreStatus.FAILED
