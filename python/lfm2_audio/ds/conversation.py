"""``Conversation`` — agrégat des tours d'un échange à l'inférence.

À ne pas confondre avec ``lfm2_audio.ds.dialogue``, qui décrit un dialogue
*d'entraînement* lu depuis un JSONL (l'audio y est un chemin de fichier).

L'agrégat porte l'invariant central du chemin multimodal : **au plus un tour
porte de l'audio**. ``multi_modal_data`` ne transporte que le signal courant ;
deux placeholders dans le prompt feraient scatter cet audio sur la position
périmée d'un tour passé, et le modèle n'entendrait plus rien au-delà du premier
tour. En le tenant ici, aucun appelant ne peut le violer par oubli.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from lfm2_audio.core.errors import PromptError
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.conversation_turn import ROLES, ConversationTurn


@dataclass(slots=True)
class Conversation:
    """Suite ordonnée de tours, garante de l'invariant « un seul audio »."""

    turns: list[ConversationTurn] = field(default_factory=list)

    def __iter__(self) -> Iterator[ConversationTurn]:
        return iter(self.turns)

    def __len__(self) -> int:
        return len(self.turns)

    @classmethod
    def from_turns(cls, turns: Sequence[ConversationTurn]) -> Conversation:
        conversation = cls(turns=list(turns))
        conversation.validate()
        return conversation

    # -- mutations ------------------------------------------------------- #

    def add(self, role: str, text: str = "", audio: Waveform | None = None) -> ConversationTurn:
        """Ajoute un tour. Retire d'abord l'audio des tours précédents."""
        if audio is not None:
            self.release_audio()
        turn = ConversationTurn(role=role, text=text, audio=audio)
        self.turns.append(turn)
        return turn

    def release_audio(self) -> None:
        """Marque tous les audios comme consommés (appelé après une génération)."""
        for turn in self.turns:
            turn.consume_audio()

    def clear(self) -> None:
        self.turns.clear()

    # -- lecture --------------------------------------------------------- #

    @property
    def pending_audio(self) -> Waveform | None:
        """Le seul audio non encore consommé, s'il existe."""
        turn = self.audio_turn
        return turn.audio if turn is not None else None

    @property
    def audio_turn(self) -> ConversationTurn | None:
        for turn in self.turns:
            if turn.has_audio:
                return turn
        return None

    def validate(self) -> None:
        """Lève ``PromptError`` si plus d'un tour porte de l'audio."""
        carrying = [turn for turn in self.turns if turn.has_audio]
        if len(carrying) > 1:
            message = (
                f"{len(carrying)} tours portent de l'audio ; un seul est permis "
                "(multi_modal_data ne transporte que l'audio courant). Appeler "
                "`release_audio()` sur les tours passés."
            )
            raise PromptError(message)


__all__ = ["ROLES", "Conversation", "ConversationTurn"]
