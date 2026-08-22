"""Tests du rendu de prompt ChatML.

Le ``Tokenizer`` étant un ``Protocol``, un double suffit : ces tests tournent
sans télécharger de modèle et sans GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from lfm2_audio.core.errors import PromptError
from lfm2_audio.core.prompt import ChatMLRenderer, RenderedPrompt, strip_special_tokens
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.conversation import ConversationTurn

AUDIO_PLACEHOLDER_ID = 128
AUDIO_TOKEN = "<|audio_start|>"


class FakeEncoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class FakeTokenizer:
    """Tokenizer minimal : conserve le texte rendu pour l'inspection."""

    def __init__(self) -> None:
        self.rendered = ""

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> FakeEncoding:
        self.rendered = text
        return FakeEncoding([ord(char) % 256 for char in text])

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        # Un marqueur est atomique ; tout autre texte se découpe par caractère.
        return [7] if text.startswith("<|") else [ord(char) for char in text]

    def decode(self, token_ids) -> str:
        return AUDIO_TOKEN if list(token_ids) == [AUDIO_PLACEHOLDER_ID] else "?"


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def renderer(tokenizer: FakeTokenizer) -> ChatMLRenderer:
    return ChatMLRenderer(tokenizer, audio_placeholder_id=AUDIO_PLACEHOLDER_ID)


def _audio() -> Waveform:
    return Waveform.of(np.zeros(1_600, dtype=np.float32), 16_000)


def test_should_open_with_bos_and_system_turn(renderer, tokenizer):
    renderer.render([ConversationTurn(role="user", text="hi")], system="Be brief.")

    assert tokenizer.rendered.startswith("<|startoftext|><|im_start|>system\nBe brief.<|im_end|>\n")


def test_should_end_with_the_assistant_prime(renderer, tokenizer):
    renderer.render([ConversationTurn(role="user", text="hi")])

    assert tokenizer.rendered.endswith("<|im_start|>assistant\n")


def test_should_emit_exactly_one_audio_placeholder(renderer, tokenizer):
    turns = [
        ConversationTurn(role="user", text="(voice message)"),
        ConversationTurn(role="assistant", text="hello"),
        ConversationTurn(role="user", audio=_audio()),
    ]

    renderer.render(turns)

    assert tokenizer.rendered.count(AUDIO_TOKEN) == 1


def test_should_attach_the_placeholder_to_the_audio_turn(renderer, tokenizer):
    renderer.render([ConversationTurn(role="user", text="now", audio=_audio())])

    assert f"<|im_start|>user\n{AUDIO_TOKEN}now<|im_end|>" in tokenizer.rendered


def test_should_carry_the_audio_into_the_rendered_prompt(renderer):
    audio = _audio()

    prompt = renderer.render([ConversationTurn(role="user", audio=audio)])

    assert prompt.audio is audio


def test_should_reject_more_than_one_audio_turn(renderer):
    turns = [
        ConversationTurn(role="user", audio=_audio()),
        ConversationTurn(role="user", audio=_audio()),
    ]

    with pytest.raises(PromptError, match="un seul est permis"):
        renderer.render(turns)


def test_text_only_prompt_should_carry_no_audio(renderer):
    prompt = renderer.render([ConversationTurn(role="user", text="hi")])

    assert prompt.audio is None
    assert "multi_modal_data" not in prompt.as_vllm_prompt()


def test_vllm_prompt_should_expose_audio_as_multimodal_data(renderer):
    audio = _audio()

    payload = renderer.render([ConversationTurn(role="user", audio=audio)]).as_vllm_prompt()

    assert payload["multi_modal_data"] == {"audio": [audio.as_model_input()]}
    assert payload["prompt_token_ids"]


def test_single_token_id_should_accept_an_atomic_marker(renderer):
    assert renderer.single_token_id("<|tool_call_end|>") == 7


def test_single_token_id_should_reject_a_split_marker(renderer):
    # Un marqueur découpé en plusieurs ids ne peut pas servir de stop token.
    with pytest.raises(PromptError, match="token unique"):
        renderer.single_token_id("not a marker")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<|text_end|>Hello<|audio_start|>", "Hello"),
        ("plain text", "plain text"),
        ("<|tool_call_start|>[f(x=1)]<|tool_call_end|>", "[f(x=1)]"),
        ("", ""),
    ],
)
def test_strip_special_tokens_should_leave_only_speakable_text(raw, expected):
    assert strip_special_tokens(raw) == expected


def test_rendered_prompt_should_copy_its_token_ids():
    # Frozen ne protège pas le contenu de la liste : `as_vllm_prompt` doit copier.
    prompt = RenderedPrompt(token_ids=[1, 2, 3])

    payload = prompt.as_vllm_prompt()
    payload["prompt_token_ids"].append(4)

    assert prompt.token_ids == [1, 2, 3]
