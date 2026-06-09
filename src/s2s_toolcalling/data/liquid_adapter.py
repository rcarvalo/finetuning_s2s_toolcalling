"""Conversion Dialogue → ``list[ChatMessage]`` liquid-audio (Phase 2).

Nécessite l'extra ``train`` (liquid-audio, torch). Implémente la convention
d'entraînement décrite dans ``dialogue_schema`` :

- tour assistant *tool call* → ``TextSegment`` seul (pas de ``<|audio_start|>``,
  l'audio est supprimé sur ce tour) ;
- tour ``tool`` → message de rôle ``tool`` (le mapper liquid-audio rend
  ``<|im_start|>{role}`` par f-string, le rôle ``tool`` passe donc tel quel
  même si le type hint amont ne le déclare pas) ;
- tour assistant final → ``InterleavedSegment`` (mode interleaved, défaut) ou
  ``TextSegment`` + ``AudioSegment`` (mode sequential).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Literal, cast

from s2s_toolcalling.data import chat_format
from s2s_toolcalling.data.dialogue_schema import Dialogue

if TYPE_CHECKING:
    from liquid_audio.data.types import ChatMessage

AssistantAudioMode = Literal["interleaved", "sequential"]


def dialogue_to_chat_messages(
    dialogue: Dialogue,
    *,
    audio_root: str | Path,
    tool_definitions: list[dict] | None = None,
    assistant_audio_mode: AssistantAudioMode = "interleaved",
    default_system: str = chat_format.DEFAULT_SYSTEM_INSTRUCTIONS,
) -> "list[ChatMessage]":
    from liquid_audio.data.types import AudioSegment, ChatMessage, InterleavedSegment, TextSegment

    audio_root = Path(audio_root)

    def read_audio(rel_path: str) -> bytes:
        return (audio_root / rel_path).read_bytes()

    messages: list[ChatMessage] = []

    system_text = chat_format.build_system_prompt(
        instructions=dialogue.system or default_system,
        tool_definitions=tool_definitions,
    )
    messages.append(ChatMessage(role="system", content=[TextSegment(text=system_text)]))

    for turn in dialogue.turns:
        if turn.role == "system":
            continue  # déjà géré ci-dessus

        if turn.role == "user":
            if turn.audio:
                content = [AudioSegment(audio=read_audio(turn.audio))]
            else:
                content = [TextSegment(text=turn.text or "")]
            messages.append(ChatMessage(role="user", content=content))

        elif turn.role == "tool":
            messages.append(
                ChatMessage(
                    # Rôle hors Literal amont mais rendu correctement par le mapper.
                    role=cast(Literal["user"], "tool"),
                    content=[TextSegment(text=chat_format.render_tool_response(turn.content))],
                )
            )

        elif turn.role == "assistant":
            if turn.tool_calls:
                calls = [(tc.name, tc.arguments) for tc in turn.tool_calls]
                messages.append(
                    ChatMessage(role="assistant", content=[TextSegment(text=chat_format.render_tool_calls(calls))])
                )
            elif turn.audio and assistant_audio_mode == "interleaved":
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=[InterleavedSegment(text=turn.text or "", audio=read_audio(turn.audio))],
                    )
                )
            elif turn.audio:  # sequential : texte puis audio
                content = []
                if turn.text:
                    content.append(TextSegment(text=turn.text))
                content.append(AudioSegment(audio=read_audio(turn.audio)))
                messages.append(ChatMessage(role="assistant", content=content))
            else:
                messages.append(ChatMessage(role="assistant", content=[TextSegment(text=turn.text or "")]))

    return messages


def dialogues_to_chat_messages(
    dialogues: Iterable[Dialogue],
    *,
    audio_root: str | Path,
    tool_definitions: list[dict] | None = None,
    assistant_audio_mode: AssistantAudioMode = "interleaved",
) -> "Iterator[list[ChatMessage]]":
    for d in dialogues:
        yield dialogue_to_chat_messages(
            d,
            audio_root=audio_root,
            tool_definitions=tool_definitions,
            assistant_audio_mode=assistant_audio_mode,
        )
