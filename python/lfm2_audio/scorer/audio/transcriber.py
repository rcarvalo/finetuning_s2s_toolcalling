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
    """Convertit un signal en texte.

    ``language`` est l'indice par appel (code ISO court, ``"fr"``/``"en"``) : un
    jeu bilingue note chaque échantillon dans SA langue au sein d'une même
    campagne. ``None`` retombe sur la langue configurée du transcripteur —
    forcer une langue unique par campagne transcrirait l'autre moitié du jeu
    en charabia et gonflerait son WER arbitrairement.
    """

    def transcribe(self, audio: Waveform, *, language: str | None = None) -> str: ...
