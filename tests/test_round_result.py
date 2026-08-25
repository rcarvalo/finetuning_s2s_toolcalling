"""``RoundResult`` — sortie d'une passe de génération, committable à part.

Ce découpage est ce qui rend la passe de décision jetable. Sans lui, une passe
qui n'émet aucun appel d'outil a déjà pollué le contexte quand on s'en aperçoit
— et l'agent doit rendre le texte séquentiel dégénéré que v3 produisait sur les
tours conversationnels.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from lfm2_audio.orchestrator.round_result import RoundResult  # noqa: E402


class _FakeChat:
    """Contrat minimal utilisé par ``commit`` : device, codebooks, append."""

    device = "cpu"
    codebooks = 8

    def __init__(self) -> None:
        self.appended: list[dict[str, Any]] = []

    def append(self, **kwargs: Any) -> None:
        self.appended.append(kwargs)


def test_should_not_touch_the_context_until_committed() -> None:
    chat = _FakeChat()
    RoundResult(visible_text="hello", text_tokens=[torch.tensor([1])], modality_flags=[0])

    assert chat.appended == []


def test_should_append_text_tokens_on_commit() -> None:
    chat = _FakeChat()
    result = RoundResult(text_tokens=[torch.tensor([7]), torch.tensor([8])], modality_flags=[0, 0])

    result.commit(chat)

    assert len(chat.appended) == 1
    assert chat.appended[0]["text"].shape == (1, 2)


def test_should_stay_silent_when_nothing_was_generated() -> None:
    # Une passe interrompue au premier token ne doit rien écrire dans le contexte.
    chat = _FakeChat()

    RoundResult().commit(chat)

    assert chat.appended == []


def test_should_pad_the_missing_modality_with_an_empty_tensor() -> None:
    chat = _FakeChat()
    result = RoundResult(text_tokens=[torch.tensor([3])], modality_flags=[0])

    result.commit(chat)

    assert chat.appended[0]["audio_out"].shape == (8, 0)


def test_should_report_whether_a_tool_call_was_emitted() -> None:
    assert RoundResult().emitted_tool_call is False
    assert RoundResult(pending_calls=[object()]).emitted_tool_call is True
