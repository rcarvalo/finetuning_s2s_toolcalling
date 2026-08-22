"""Tests de l'agrégat ``Conversation`` et de son invariant « un seul audio ».

C'est la régression du bug multi-tours : plusieurs placeholders audio dans un
prompt pour un seul signal faisaient scatter l'audio courant sur la position
périmée d'un tour passé, et le modèle n'entendait plus rien au-delà du tour 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from lfm2_audio.core.errors import PromptError
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.conversation import Conversation, ConversationTurn


def _audio() -> Waveform:
    return Waveform.of(np.zeros(1_600, dtype=np.float32), 16_000)


def test_should_reject_an_unknown_role():
    with pytest.raises(ValueError, match="role"):
        ConversationTurn(role="wizard", text="hi")


def test_turn_should_report_whether_it_carries_audio():
    assert ConversationTurn(role="user", audio=_audio()).has_audio
    assert not ConversationTurn(role="user", text="hi").has_audio


def test_consume_audio_should_drop_signal_and_leave_a_textual_trace():
    turn = ConversationTurn(role="user", audio=_audio())

    turn.consume_audio()

    assert turn.audio is None
    assert turn.text == "(voice message)"


def test_consume_audio_should_preserve_existing_text():
    turn = ConversationTurn(role="user", text="what's the weather", audio=_audio())

    turn.consume_audio()

    assert turn.text == "what's the weather"


def test_adding_audio_should_release_the_previous_one():
    conversation = Conversation()
    conversation.add("user", audio=_audio())
    conversation.add("assistant", text="first answer")

    conversation.add("user", audio=_audio())

    carrying = [turn for turn in conversation if turn.has_audio]
    assert len(carrying) == 1
    assert carrying[0] is conversation.turns[-1]


def test_release_audio_should_clear_every_signal():
    conversation = Conversation()
    conversation.add("user", audio=_audio())

    conversation.release_audio()

    assert conversation.pending_audio is None
    assert conversation.audio_turn is None


def test_validate_should_reject_two_audio_turns():
    # Construction directe : contourne `add`, comme le ferait un appelant externe.
    turns = [
        ConversationTurn(role="user", audio=_audio()),
        ConversationTurn(role="user", audio=_audio()),
    ]

    with pytest.raises(PromptError, match="un seul est permis"):
        Conversation.from_turns(turns)


def test_from_turns_should_accept_a_single_audio_turn():
    turns = [
        ConversationTurn(role="user", text="(voice message)"),
        ConversationTurn(role="assistant", text="hello"),
        ConversationTurn(role="user", audio=_audio()),
    ]

    conversation = Conversation.from_turns(turns)

    assert len(conversation) == 3
    assert conversation.pending_audio is not None


def test_clear_should_empty_the_history():
    conversation = Conversation()
    conversation.add("user", text="hi")

    conversation.clear()

    assert len(conversation) == 0
