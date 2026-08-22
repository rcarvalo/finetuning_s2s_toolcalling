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

Modèles **pydantic** : ces dialogues viennent d'un fichier écrit par un autre
outil (génération LLM, TTS), donc d'une frontière externe — c'est exactement
là que la validation doit vivre. La conversion vers liquid-audio est dans
``lfm2_audio.data_prep.liquid_adapter``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant", "tool"]

VALID_ROLES: tuple[str, ...] = ("system", "user", "assistant", "tool")


class DialogueValidationError(ValueError):
    """Ligne JSONL non conforme au schéma des dialogues d'entraînement."""


class ToolCall(BaseModel):
    """Appel d'outil attendu de l'assistant."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class DialogueMeta(BaseModel):
    """Métadonnées de génération d'un dialogue synthétique.

    Alimente la QA du dataset (distribution des styles, des cibles d'outil) sans
    entrer dans le prompt d'entraînement. ``extra="allow"`` : un générateur peut
    y ajouter ses propres clés sans invalider les datasets existants.
    """

    model_config = ConfigDict(extra="allow")

    style: str | None = None
    depth: str | None = None
    target: str | None = None


class Turn(BaseModel):
    """Un tour du dialogue.

    ``audio`` est un chemin **relatif** à la racine audio du dataset — à ne pas
    confondre avec :class:`lfm2_audio.ds.conversation.ConversationTurn`, qui
    porte le signal lui-même à l'inférence.
    """

    model_config = ConfigDict(extra="forbid")

    role: Role
    text: str | None = None
    audio: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    content: Any = None
    """Payload d'un tour ``tool`` (résultat d'exécution réinjecté)."""

    voice: str | None = None
    """TTS voice this turn was synthesized with.

    Written by ``lfm2-synthesize-audio`` and by the Hub rehydration, and read
    back when analysing whether a model generalizes across timbres. Declared
    here because ``extra="forbid"`` otherwise rejects our own pipeline's output.
    """

    @model_validator(mode="after")
    def _check_role_invariants(self) -> Self:
        """Contraintes de la stratégie « penser en texte, parler en audio »."""
        if self.role == "assistant" and self.tool_calls and self.audio:
            message = "un tour de tool call ne doit pas porter d'audio (les tool calls sont émis en texte seul)"
            raise ValueError(message)
        if self.role == "tool" and self.content is None:
            message = "un tour `tool` doit porter 'content'"
            raise ValueError(message)
        if self.role == "user" and not (self.text or self.audio):
            message = "un tour `user` doit porter 'text' ou 'audio'"
            raise ValueError(message)
        if self.role == "assistant" and not self.tool_calls and not (self.text or self.audio):
            message = "un tour `assistant` sans tool call ne peut pas être vide"
            raise ValueError(message)
        return self

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)


class Dialogue(BaseModel):
    """Un dialogue complet — une ligne du JSONL d'entraînement."""

    model_config = ConfigDict(extra="forbid")

    id: str
    turns: list[Turn]
    system: str | None = None
    tools: list[str] = Field(default_factory=list)
    meta: DialogueMeta = Field(default_factory=DialogueMeta)

    @property
    def has_tool_call(self) -> bool:
        """Vrai si au moins un tour assistant appelle un outil (vs. un négatif)."""
        return any(turn.is_tool_call for turn in self.turns)


def parse_dialogue(obj: dict[str, Any]) -> Dialogue:
    """Valide un dict et le convertit en :class:`Dialogue`.

    Traduit les ``ValidationError`` de pydantic en
    :class:`DialogueValidationError`, pour que les appelants n'aient qu'un seul
    type d'erreur à connaître.
    """
    try:
        return Dialogue.model_validate(obj)
    except ValueError as exc:
        identifier = obj.get("id", "<sans id>")
        message = f"{identifier}: {exc}"
        raise DialogueValidationError(message) from exc


def load_dialogues(path: str | Path) -> Iterator[Dialogue]:
    """Itère les dialogues d'un fichier JSONL (validation à la volée)."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                message = f"{path}:{line_no}: JSON invalide : {exc}"
                raise DialogueValidationError(message) from exc
            yield parse_dialogue(obj)
