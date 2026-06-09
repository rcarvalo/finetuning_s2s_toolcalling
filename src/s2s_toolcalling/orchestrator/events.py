"""Événements émis par l'agent vers le transport (WebSocket / fastrtc / Reachy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TextDelta:
    """Texte « visible » incrémental (inner monologue sans les spans de tool call)."""

    text: str


@dataclass(slots=True)
class AudioChunk:
    """PCM float32 mono 24 kHz (1920 échantillons = 80 ms par frame Mimi)."""

    samples: Any  # np.ndarray | torch.Tensor (le transport convertit)
    sample_rate: int = 24_000


@dataclass(slots=True)
class FillerSpeech:
    """Filler vocal à jouer pendant le round-trip de l'outil (« je vérifie... »)."""

    phrase: str
    wav_path: str | None = None


@dataclass(slots=True)
class ToolCallBegin:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolCallResult:
    name: str
    ok: bool
    payload: Any
    elapsed_ms: float


@dataclass(slots=True)
class TurnComplete:
    """Fin du tour assistant : texte final + stats."""

    text: str
    tool_rounds: int = 0
    audio_frames: int = 0


@dataclass(slots=True)
class AgentError:
    message: str
    recoverable: bool = True


AgentEvent = TextDelta | AudioChunk | FillerSpeech | ToolCallBegin | ToolCallResult | TurnComplete | AgentError


def event_to_dict(event: AgentEvent) -> dict[str, Any]:
    """Sérialisation JSON-compatible pour le transport WebSocket."""
    if isinstance(event, AudioChunk):
        return {"type": "audio_chunk", "sample_rate": event.sample_rate}
    mapping = {
        TextDelta: "text_delta",
        FillerSpeech: "filler",
        ToolCallBegin: "tool_call_begin",
        ToolCallResult: "tool_call_result",
        TurnComplete: "turn_complete",
        AgentError: "error",
    }
    out: dict[str, Any] = {"type": mapping[type(event)]}
    for slot in event.__dataclass_fields__:
        out[slot] = getattr(event, slot)
    return out
