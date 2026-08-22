"""``QuestionSet`` — le jeu d'évaluation, chargé depuis un JSONL."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from lfm2_audio.ds.dialogue import load_dialogues
from lfm2_audio.evaluation.question import Question


@dataclass(frozen=True, slots=True)
class QuestionSet:
    """Suite ordonnée de questions, avec sa provenance.

    ``source`` est conservée pour que le rapport dise sur quoi il a été produit :
    comparer deux campagnes sur des jeux différents est l'erreur la plus facile à
    commettre et la plus difficile à repérer après coup.
    """

    questions: tuple[Question, ...]
    source: str = ""
    audio_root: Path | None = None

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions)

    def __len__(self) -> int:
        return len(self.questions)

    @classmethod
    def from_jsonl(cls, path: str | Path, *, audio_root: str | Path | None = None) -> Self:
        """Charge un JSONL au schéma des dialogues (validation pydantic incluse)."""
        root = Path(audio_root) if audio_root else None
        questions = tuple(Question.from_dialogue(dialogue, audio_root=root) for dialogue in load_dialogues(path))
        return cls(questions=questions, source=str(path), audio_root=root)

    def take(self, limit: int | None) -> Self:
        """Sous-ensemble des ``limit`` premières questions (tout si ``None``)."""
        if limit is None or limit >= len(self.questions):
            return self
        return type(self)(questions=self.questions[:limit], source=self.source, audio_root=self.audio_root)

    def filter_ids(self, ids: Sequence[str]) -> Self:
        wanted = set(ids)
        return type(self)(
            questions=tuple(q for q in self.questions if q.question_id in wanted),
            source=self.source,
            audio_root=self.audio_root,
        )

    @property
    def positives(self) -> int:
        """Questions attendant un appel d'outil."""
        return sum(q.expects_tool_call for q in self.questions)
