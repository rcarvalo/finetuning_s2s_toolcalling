"""`Turn.voice` — our own pipeline writes it, so the schema must accept it.

Regression: `Turn` is `extra="forbid"` and did not declare `voice`, while both
`lfm2-synthesize-audio` and the Hub rehydration write it on user turns. Every
rehydrated dialogue was therefore rejected, blocking packing and evaluation
alike.
"""

from __future__ import annotations

import pytest

from lfm2_audio.ds.dialogue import DialogueValidationError, parse_dialogue


def _dialogue(**turn_extra: object) -> dict:
    return {"id": "d1", "tools": [], "turns": [{"role": "user", "text": "hi", "audio": "a.wav", **turn_extra}]}


def test_should_accept_a_voice_on_a_turn() -> None:
    dialogue = parse_dialogue(_dialogue(voice="casual_female"))

    assert dialogue.turns[0].voice == "casual_female"


def test_should_default_voice_to_none() -> None:
    assert parse_dialogue(_dialogue()).turns[0].voice is None


def test_should_still_reject_an_unknown_field() -> None:
    """The point of extra=forbid is to catch typos; voice is the only addition."""
    with pytest.raises(DialogueValidationError):
        parse_dialogue(_dialogue(voiec="casual_female"))
