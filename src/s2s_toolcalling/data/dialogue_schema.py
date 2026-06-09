"""Schéma JSONL des dialogues d'entraînement (Phase 2b — SFT interleaved).

Chaque ligne du JSONL est un dialogue :

.. code-block:: json

    {
      "id": "dlg_0001",
      "system": "Tu es l'assistant vocal d'accueil...",
      "tools": ["check_appointment", "notify_employee"],
      "turns": [
        {"role": "user", "text": "Bonjour, j'ai rendez-vous avec Mme Martin.",
         "audio": "audio/dlg_0001_u0.wav"},
        {"role": "assistant",
         "tool_calls": [{"name": "check_appointment",
                         "arguments": {"visitor_name": "M. Petit", "host_name": "Mme Martin"}}]},
        {"role": "tool", "content": {"found": true, "time": "14:00", "room": "B2"}},
        {"role": "assistant", "text": "Oui, vous êtes attendu à 14 heures en salle B2.",
         "audio": "audio/dlg_0001_a1.wav"}
      ]
    }

Conventions d'entraînement (stratégie « thinking in text, speaking in audio ») :

- tour **user** : audio (AudioSegment) si ``audio`` présent, sinon texte ;
- tour **assistant avec tool_calls** : texte SEUL (l'audio est supprimé — le
  modèle apprend à émettre le tool call dans le flux texte puis ``<|im_end|>``) ;
- tour **tool** : texte seul (résultat JSON réinjecté par l'orchestrateur) ;
- tour **assistant final** : texte + audio (InterleavedSegment en mode interleaved).

Module Python pur — la conversion vers liquid-audio est dans ``liquid_adapter``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

VALID_ROLES = ("system", "user", "assistant", "tool")


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Turn:
    role: str
    text: str | None = None
    audio: str | None = None  # chemin relatif au root audio
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: Any = None  # payload des tours `tool`


@dataclass(slots=True)
class Dialogue:
    id: str
    turns: list[Turn]
    system: str | None = None
    tools: list[str] = field(default_factory=list)


class DialogueValidationError(ValueError):
    pass


def parse_dialogue(obj: dict[str, Any]) -> Dialogue:
    if "id" not in obj or "turns" not in obj:
        raise DialogueValidationError("dialogue must have 'id' and 'turns'")

    turns: list[Turn] = []
    for i, t in enumerate(obj["turns"]):
        role = t.get("role")
        if role not in VALID_ROLES:
            raise DialogueValidationError(f"{obj['id']}: turn {i} has invalid role {role!r}")

        tool_calls = [ToolCall(name=tc["name"], arguments=tc.get("arguments", {})) for tc in t.get("tool_calls", [])]

        if role == "assistant" and tool_calls and t.get("audio"):
            raise DialogueValidationError(
                f"{obj['id']}: turn {i} is a tool-call turn and must not carry audio "
                "(tool calls are emitted text-only)"
            )
        if role == "tool" and t.get("content") is None:
            raise DialogueValidationError(f"{obj['id']}: turn {i} (role=tool) must have 'content'")
        if role == "user" and not (t.get("text") or t.get("audio")):
            raise DialogueValidationError(f"{obj['id']}: turn {i} (role=user) needs 'text' or 'audio'")
        if role == "assistant" and not tool_calls and not (t.get("text") or t.get("audio")):
            raise DialogueValidationError(f"{obj['id']}: turn {i} (role=assistant) is empty")

        turns.append(
            Turn(
                role=role,
                text=t.get("text"),
                audio=t.get("audio"),
                tool_calls=tool_calls,
                content=t.get("content"),
            )
        )

    return Dialogue(
        id=str(obj["id"]),
        turns=turns,
        system=obj.get("system"),
        tools=list(obj.get("tools", [])),
    )


def load_dialogues(path: str | Path) -> Iterator[Dialogue]:
    """Itère les dialogues d'un fichier JSONL (validation à la volée)."""
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DialogueValidationError(f"{path}:{line_no}: invalid JSON: {e}") from e
            yield parse_dialogue(obj)
