"""``ScorerFactory`` — construit les scorers d'une :class:`ScoringConfig`.

C'est ici que vivent les dépendances partagées : le WER a besoin d'un
transcripteur, le raisonnement d'un juge, et les deux coûtent cher à charger.
La fabrique les instancie **une fois** et les injecte, plutôt que de laisser
chaque scorer se débrouiller — trois scorers audio ne doivent pas charger trois
Whisper.

Un scorer dont les dépendances manquent n'est pas une erreur : il est remplacé
par un :class:`MissingScorer` qui rend ``UNAVAILABLE`` avec la raison, sauf si
la config demande explicitement l'échec.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.core.lazy_component import LazyComponent
from lfm2_audio.ds.scoring_config import ScorerConfig, ScoringConfig
from lfm2_audio.scorer.audio.transcriber import Transcriber
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.missing import MissingScorer
from lfm2_audio.scorer.registry import SCORERS, ScorerRegistry
from lfm2_audio.scorer.text.judge import Judge

logger = logging.getLogger(__name__)

_NEEDS_TRANSCRIBER = frozenset({"wer"})
_NEEDS_JUDGE = frozenset({"reasoning"})

TRANSCRIBER = LazyComponent(
    module="lfm2_audio.scorer.audio.whisper_transcriber",
    class_name="WhisperTranscriber",
    requires=("torch", "transformers"),
)
# ASR CPU (CTranslate2, int8) : le chemin par défaut charge Whisper sur
# cuda:0, ce qui fait déborder un GPU déjà occupé par l'entraînement ou par le
# modèle audio de la campagne. Sélectionné par `asr_backend: faster_whisper`.
FASTER_WHISPER = LazyComponent(
    module="lfm2_audio.scorer.audio.faster_whisper_transcriber",
    class_name="FasterWhisperTranscriber",
    requires=("faster_whisper",),
)
JUDGE = LazyComponent(
    module="lfm2_audio.scorer.text.gemini_judge",
    class_name="GeminiJudge",
    requires=("google.genai",),
)
# Repli quand GEMINI_API_KEY est absent : le juge est un Protocol, la source du
# jugement est donc interchangeable sans toucher scorer ni rubrique. Une
# campagne enregistre le modèle de juge utilisé — deux campagnes jugées par des
# modèles différents ne sont pas comparables.
HF_JUDGE = LazyComponent(
    module="lfm2_audio.scorer.text.hf_judge",
    class_name="HfJudge",
    requires=("huggingface_hub",),
)


class ScorerFactory:
    """Assemble des scorers prêts à l'emploi à partir d'une configuration."""

    def __init__(self, config: ScoringConfig, *, registry: ScorerRegistry = SCORERS) -> None:
        self._config = config
        self._registry = registry
        self._transcriber: Transcriber | None = None
        self._judge: Judge | None = None

    def build_all(self) -> list[BaseScorer]:
        """Tous les scorers activés de la config, dans l'ordre déclaré."""
        return [self.build(entry) for entry in self._config.scorers if entry.enabled]

    def build(self, entry: ScorerConfig) -> BaseScorer:
        """Un scorer, ou son substitut indisponible."""
        spec = self._registry.describe(entry.name)

        reason = spec.unavailable_reason()
        if reason is not None:
            if self._config.fail_on_unavailable:
                message = f"scorer {entry.name!r} indisponible : {reason}"
                raise Lfm2AudioError(message)
            logger.warning("scorer %s indisponible : %s", entry.name, reason)
            return MissingScorer(entry.name, reason)

        options = dict(entry.options)
        self._inject_shared_dependencies(entry.name, options)
        return spec.load()(**options)

    # ------------------------------------------------------------------ #

    def _inject_shared_dependencies(self, name: str, options: dict[str, Any]) -> None:
        """Ajoute transcripteur et juge aux kwargs, sans écraser un choix explicite."""
        if name in _NEEDS_TRANSCRIBER and "transcriber" not in options:
            options["transcriber"] = self._shared_transcriber()
        if name in _NEEDS_JUDGE and "judge" not in options:
            options["judge"] = self._shared_judge()

    def _shared_transcriber(self) -> Transcriber:
        """Whisper, construit une seule fois pour tous les scorers qui transcrivent."""
        if self._transcriber is None:
            if self._config.asr_backend == "faster_whisper":
                self._transcriber = FASTER_WHISPER.build(
                    model_size=self._config.asr_model_size,
                    language=self._config.asr_language,
                )
            else:
                self._transcriber = TRANSCRIBER.build(
                    model_id=self._config.asr_model_id,
                    device=self._config.asr_device,
                    language=self._config.asr_language,
                )
        return self._transcriber

    def _shared_judge(self) -> Judge:
        """Juge LLM, construit une seule fois.

        Gemini d'abord (clé historique du projet), repli Hugging Face sinon :
        sans repli, l'absence de `GEMINI_API_KEY` rend le scorer `reasoning`
        indisponible et la campagne mesure l'ancrage sans jamais mesurer la
        pertinence — exactement l'angle mort de la validation v3.
        """
        if self._judge is None:
            if os.environ.get("GEMINI_API_KEY"):
                self._judge = JUDGE.build(model_id=self._config.judge_model_id)
            else:
                self._judge = HF_JUDGE.build()
        return self._judge
