"""``WhisperTranscriber`` — ASR de référence pour le WER, via transformers.

Whisper est le standard de fait pour mesurer l'intelligibilité d'un TTS : on
transcrit l'audio généré et on compare à ce que le modèle était censé dire.

Ce module importe torch et transformers **en tête** : il est donc lourd, et
n'est importé que par qui veut réellement Whisper. Le registre de scorers le
résout par chaîne, si bien qu'une machine sans torch peut lister les scorers
sans le charger.

Le *modèle*, lui, reste chargé paresseusement au premier appel : construire un
transcripteur dans une config d'entraînement ne doit pas coûter 3 Go de VRAM
tant qu'aucune évaluation ne tourne.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import pipeline

from lfm2_audio.ds.audio import INPUT_SAMPLE_RATE, Waveform

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "openai/whisper-large-v3-turbo"


class WhisperTranscriber:
    """Transcripteur Whisper. Satisfait le protocole ``Transcriber``."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str | None = None,
        language: str = "en",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._language = language
        self._pipeline: Any = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def transcribe(self, audio: Waveform) -> str:
        """Transcrit un signal. Whisper attend du 16 kHz mono."""
        resampled = audio.resample(INPUT_SAMPLE_RATE)
        output = self._asr()(
            {"raw": resampled.samples, "sampling_rate": resampled.sample_rate},
            generate_kwargs={"language": self._language},
        )
        return str(output["text"]).strip()

    def _asr(self) -> Any:  # noqa: ANN401 — modèle tiers non typé
        """Pipeline transformers, construit au premier usage puis conservé."""
        if self._pipeline is None:
            device = self._device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            logger.info("chargement de %s sur %s", self._model_id, device)
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self._model_id,
                device=device,
                torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            )
        return self._pipeline
