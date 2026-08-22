"""``Question`` — un cas du jeu d'évaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from lfm2_audio.ds.dialogue import Dialogue


@dataclass(frozen=True, slots=True)
class Question:
    """Ce qu'on demande au modèle, et ce qu'on attend de lui.

    Le jeu de questions et le jeu d'entraînement partagent le schéma JSONL des
    dialogues : un même fichier peut donc servir au TTS puis à l'évaluation,
    sans conversion intermédiaire.
    """

    question_id: str
    text: str = ""
    audio_path: Path | None = None
    expected_calls: list[dict[str, Any]] = field(default_factory=list)
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expects_tool_call(self) -> bool:
        return bool(self.expected_calls)

    @classmethod
    def from_dialogue(cls, dialogue: Dialogue, *, audio_root: Path | None = None) -> Self:
        """Extrait le tour user et les attentes du tour assistant."""
        user = next((t for t in dialogue.turns if t.role == "user"), None)
        assistant = next((t for t in dialogue.turns if t.role == "assistant"), None)

        audio = None
        if user is not None and user.audio:
            audio = Path(audio_root or ".") / user.audio

        return cls(
            question_id=dialogue.id,
            text=(user.text or "") if user else "",
            audio_path=audio,
            expected_calls=[
                {"name": c.name, "arguments": c.arguments} for c in (assistant.tool_calls if assistant else [])
            ],
            reference_answer=(assistant.text or "") if assistant else "",
            metadata=dialogue.meta.model_dump(exclude_none=True),
        )
