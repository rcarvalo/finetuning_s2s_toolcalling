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
from typing import Any

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.ds.scoring_config import ScorerConfig, ScoringConfig
from lfm2_audio.scorer.audio.transcriber import Transcriber
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.lazy import LazyComponent
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
JUDGE = LazyComponent(
    module="lfm2_audio.scorer.text.gemini_judge",
    class_name="GeminiJudge",
    requires=("google.genai",),
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
            self._transcriber = TRANSCRIBER.build(model_id=self._config.asr_model_id)
        return self._transcriber

    def _shared_judge(self) -> Judge:
        """Juge LLM, construit une seule fois."""
        if self._judge is None:
            self._judge = JUDGE.build(model_id=self._config.judge_model_id)
        return self._judge
