"""``FasterWhisperTranscriber`` — l'ASR du WER, sur CPU, sans toucher au GPU.

Le transcripteur ``transformers`` par défaut charge Whisper sur ``cuda:0``.
Or le GPU est déjà pris : l'entraînement culmine à ~96 % de la VRAM d'une L4,
et une campagne d'évaluation y tient déjà le modèle audio. Ajouter Whisper à
côté fait déborder l'un ou l'autre.

faster-whisper (CTranslate2, int8) transcrit sur CPU à une vitesse utilisable
et ne tire pas torch. Le WER devient donc mesurable **pendant** un
entraînement, et sur un portable — ce que le chemin GPU interdisait.

Satisfait le ``Protocol`` :class:`Transcriber`, comme le transcripteur
``transformers`` : le scorer ne connaît ni l'un ni l'autre.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lfm2_audio.ds.audio import INPUT_SAMPLE_RATE, Waveform

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"


class FasterWhisperTranscriber:
    """Transcription CPU d'un signal généré, chargée au premier usage."""

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        *,
        language: str | None = "en",
        compute_type: str = "int8",
        device: str = "cpu",
    ) -> None:
        self._model_size = model_size
        self._language = language
        self._compute_type = compute_type
        self._device = device
        self._model: WhisperModel | None = None

    @property
    def model_id(self) -> str:
        return f"faster-whisper/{self._model_size}"

    def transcribe(self, audio: Waveform, *, language: str | None = None) -> str:
        """Texte du signal, ou chaîne vide si faster-whisper n'est pas installé.

        Une dépendance absente rend le WER ``UNAVAILABLE`` en amont ; renvoyer
        une chaîne vide ici produirait un WER de 1.0 qu'on lirait comme une
        mesure — c'est pour ça que l'absence est signalée par le scorer, pas
        déguisée en score.
        """
        model = self._load()
        if model is None:
            return ""
        segments, _ = model.transcribe(
            audio.resample(INPUT_SAMPLE_RATE).samples,
            language=language or self._language,
            beam_size=1,
        )
        return " ".join(segment.text for segment in segments).strip()

    def _load(self) -> WhisperModel | None:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                logger.warning("faster-whisper absent : pas de WER (uv sync --extra eval)")
                return None
            logger.info("chargement de whisper-%s sur %s (%s)", self._model_size, self._device, self._compute_type)
            self._model = WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)
        return self._model
