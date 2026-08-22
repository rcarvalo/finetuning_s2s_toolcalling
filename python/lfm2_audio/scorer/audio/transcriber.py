"""``Transcriber`` — contrat ASR minimal attendu par le WER.

Un ``Protocol`` plutôt qu'une dépendance en dur : le WER dépend du *fait* de
transcrire, pas de Whisper. Un double de test satisfait ce contrat sans GPU, et
brancher un autre ASR ne touche pas le scorer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lfm2_audio.ds.audio import Waveform


@runtime_checkable
class Transcriber(Protocol):
    """Convertit un signal en texte."""

    def transcribe(self, audio: Waveform) -> str: ...
