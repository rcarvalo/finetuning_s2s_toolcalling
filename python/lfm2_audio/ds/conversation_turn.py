"""``ConversationTurn`` — un tour d'un échange à l'inférence.

À ne pas confondre avec :class:`lfm2_audio.ds.dialogue.Turn`, qui décrit un
tour *d'entraînement* lu depuis un JSONL (l'audio y est un chemin de fichier).
"""

from __future__ import annotations

from dataclasses import dataclass

from lfm2_audio.ds.audio import Waveform

ROLES = ("system", "user", "assistant", "tool")

_VOICE_PLACEHOLDER = "(voice message)"


@dataclass(slots=True)
class ConversationTurn:
    """Un tour. Mutable : l'audio est retiré une fois consommé par le modèle."""

    role: str
    text: str = ""
    audio: Waveform | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            message = f"role doit être parmi {ROLES}, reçu {self.role!r}"
            raise ValueError(message)

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    def consume_audio(self) -> None:
        """Retire l'audio après usage, en gardant une trace textuelle du tour.

        Le signal n'existe plus pour les passes suivantes : le conserver
        produirait un placeholder sans frame réelle derrière.
        """
        if self.audio is not None:
            self.audio = None
            self.text = self.text or _VOICE_PLACEHOLDER
