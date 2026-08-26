"""Tests of the ASR benchmark selection logic (no network, no audio)."""

from __future__ import annotations

from lfm2_audio.data_prep.asr_bench import AsrCandidate, AsrClipSelector, asr_dialogue


def _candidate(i: int, speaker: str = "spk", score: float | None = None) -> AsrCandidate:
    return AsrCandidate(sample_id=f"c{i}", transcript=f"phrase {i}", speaker=speaker, score=score)


def test_should_accept_until_the_limit() -> None:
    selector = AsrClipSelector(limit=2)

    assert selector.offer(_candidate(1)) is True
    assert selector.offer(_candidate(2)) is True
    assert selector.offer(_candidate(3)) is False
    assert selector.full is True
    assert selector.accepted == 2


def test_should_reject_below_the_quality_floor() -> None:
    selector = AsrClipSelector(limit=10, min_score=3.5)

    assert selector.offer(_candidate(1, score=3.4)) is False
    assert selector.offer(_candidate(2, score=None)) is False
    assert selector.offer(_candidate(3, score=3.5)) is True


def test_should_cap_clips_per_speaker() -> None:
    selector = AsrClipSelector(limit=10, max_per_speaker=2)

    assert selector.offer(_candidate(1, speaker="a")) is True
    assert selector.offer(_candidate(2, speaker="a")) is True
    assert selector.offer(_candidate(3, speaker="a")) is False
    assert selector.offer(_candidate(4, speaker="b")) is True


def test_should_not_cap_an_unknown_speaker() -> None:
    """An empty speaker id must not make all anonymous clips share one quota."""
    selector = AsrClipSelector(limit=10, max_per_speaker=1)

    assert selector.offer(_candidate(1, speaker="")) is True
    assert selector.offer(_candidate(2, speaker="")) is True


def test_should_reject_an_empty_transcript() -> None:
    selector = AsrClipSelector(limit=10)

    candidate = AsrCandidate(sample_id="c1", transcript="   ")

    assert selector.offer(candidate) is False


def test_should_record_the_selected_speakers() -> None:
    selector = AsrClipSelector(limit=10)
    selector.offer(_candidate(1, speaker="a"))
    selector.offer(_candidate(2, speaker="b"))
    selector.offer(_candidate(3, speaker=""))

    assert selector.speakers == {"a", "b"}


def test_asr_dialogue_should_carry_lang_and_reference() -> None:
    case = asr_dialogue("fleurs_fr_00001", "bonjour à tous", "fleurs_fr_00001.wav", "fr")

    assert case["meta"] == {"lang": "fr", "task": "asr"}
    assert case["turns"][0] == {"role": "user", "audio": "fleurs_fr_00001.wav"}
    assert case["turns"][1] == {"role": "assistant", "text": "bonjour à tous"}
