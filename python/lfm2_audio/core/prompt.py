"""Rendu du prompt ChatML LFM2.5 pour l'inférence (source unique).

Ce module concentre la **politique de placeholder audio**, qui était auparavant
recopiée dans chaque script de démo — et c'est précisément là que vivait le bug
multi-tours : N placeholders émis pour UN seul audio faisaient scatter le signal
courant sur la position périmée du premier tour, et le modèle n'« entendait »
plus rien au-delà du tour 1.

Règle : ``multi_modal_data`` ne porte QUE l'audio du tour courant, donc le prompt
ne doit contenir **exactement un** placeholder. Un seul tour peut porter de
l'audio ; les tours audio passés sont rendus en texte seul.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lfm2_audio.core.chat_format import TOOL_CALL_SPAN, TOOL_CALL_START
from lfm2_audio.core.errors import PromptError
from lfm2_audio.core.tokenizer import Tokenizer
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.conversation import ConversationTurn

BOS = "<|startoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Tous les marqueurs <|…|> (audio_start, text_end, tool_call_*, …). À retirer du
# texte PARLABLE et de l'historique : re-tokenisés, les <|audio_start|> orphelins
# (sans frame réelle derrière) corrompent le contexte du tour suivant.
_SPECIAL_TOKEN = re.compile(r"<\|[^|>]*\|>")

DEFAULT_SYSTEM = "Respond with interleaved text and audio."

BILINGUAL_SYSTEM = (
    "You are a bilingual voice assistant. Always reply in the same language the user speaks. "
    "Tu es un assistant vocal bilingue. Réponds toujours dans la langue que l'utilisateur emploie."
)
"""Retained after a four-arm measurement on ``lang_mirror`` (2026-08-27).

French mirroring, from the base model, by system prompt: 40 % with the default,
79 % with an English rule, 90 % here, 94 % with a plain French sentence and no
rule at all. This one is kept over the 94 % arm because it wins on code-switch
(95 % vs 90 %) — following a user who changes language mid-conversation is the
use case, not an edge case — and because an explicit bilingual instruction is
something a human can read and revise, where "it works because the prompt is in
French" is a side effect nobody controls.

Mirror rate is not French QUALITY: replies still inject English nouns
mid-sentence. What the prompt fixes is which language the model aims for.
"""

SYSTEM_PROMPTS = {"default": DEFAULT_SYSTEM, "bilingual": BILINGUAL_SYSTEM}
"""Named prompts, so a campaign can be reproduced by name.

Two campaigns are only comparable at identical system prompt, so the prompt is
a frozen parameter like the judge's rubric version. Naming it also keeps it off
the command line, where Inspect splits ``-M`` values on commas and would tear a
sentence in half.
"""


def resolve_system(system: str | None) -> str | None:
    """A named prompt, or the text itself when it is not a known name."""
    if system is None:
        return None
    return SYSTEM_PROMPTS.get(system, system)


def strip_special_tokens(text: str) -> str:
    """Texte « parlable » : tous les marqueurs spéciaux retirés."""
    return _SPECIAL_TOKEN.sub("", text).strip()


def spoken_part(text: str) -> str:
    """What the model actually said: the whole call span removed, not just its markers.

    Stripping the markers alone would leave ``[web_search(query="…")]`` looking
    like a sentence — which is how a tool call ends up being judged as speech,
    or shown to a reader as the spoken answer. An unterminated span (vLLM strips
    the stop token) is cut to the end for the same reason.
    """
    without_calls = TOOL_CALL_SPAN.sub(" ", text)
    if TOOL_CALL_START in without_calls:
        without_calls = without_calls[: without_calls.index(TOOL_CALL_START)]
    return strip_special_tokens(without_calls)


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Prompt prêt pour ``Omni.generate``."""

    token_ids: list[int]
    audio: Waveform | None = None

    def as_vllm_prompt(self) -> dict[str, Any]:
        """Dict attendu par vLLM (``multi_modal_data`` seulement s'il y a de l'audio)."""
        prompt: dict[str, Any] = {"prompt_token_ids": list(self.token_ids)}
        if self.audio is not None:
            # vLLM attend le tuple (signal, fréquence), pas le value object.
            prompt["multi_modal_data"] = {"audio": [self.audio.as_model_input()]}
        return prompt


class ChatMLRenderer:
    """Tours → ids de prompt ChatML, avec au plus un placeholder audio.

    ``audio_placeholder_id`` est l'id réservé du plugin pour une frame audio ; le
    processor multimodal le remplace par ``ceil(T_mel / 8)`` placeholders réels
    au prefill (cf. ``docs/audio_in_spec.md``).
    """

    def __init__(self, tokenizer: Tokenizer, *, audio_placeholder_id: int) -> None:
        self._tokenizer = tokenizer
        self._audio_token = tokenizer.decode([audio_placeholder_id])

    def render(self, turns: Sequence[ConversationTurn], *, system: str = DEFAULT_SYSTEM) -> RenderedPrompt:
        """Rend les tours + l'amorce ``<|im_start|>assistant``.

        Lève ``PromptError`` si plus d'un tour porte de l'audio : vLLM ne reçoit
        qu'un audio dans ``multi_modal_data``, deux placeholders le feraient
        scatter à la mauvaise position.
        """
        audio_turns = [turn for turn in turns if turn.has_audio]
        if len(audio_turns) > 1:
            message = (
                f"{len(audio_turns)} tours portent de l'audio ; un seul est permis "
                "(multi_modal_data ne transporte que l'audio courant). Remettre "
                "`audio=None` sur les tours passés."
            )
            raise PromptError(message)

        parts = [f"{BOS}{IM_START}system\n{system}{IM_END}\n"]
        for turn in turns:
            text = turn.text or ""
            if turn.has_audio:
                text = f"{self._audio_token}{text}"
            parts.append(f"{IM_START}{turn.role}\n{text}{IM_END}\n")
        parts.append(f"{IM_START}assistant\n")

        token_ids = self._tokenizer("".join(parts), add_special_tokens=False).input_ids
        audio = audio_turns[0].audio if audio_turns else None
        return RenderedPrompt(token_ids=list(token_ids), audio=audio)

    def single_token_id(self, token: str) -> int:
        """Id d'un token spécial, en vérifiant qu'il est bien atomique.

        Un marqueur découpé en plusieurs ids ne peut pas servir de ``stop_token``.
        """
        ids = self._tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            message = f"{token!r} doit être un token unique, obtenu {ids}"
            raise PromptError(message)
        return int(ids[0])
