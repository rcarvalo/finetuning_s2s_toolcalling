"""``ScoringConfig`` — quels scorers activer, et avec quels réglages.

Modèle pydantic : cette configuration vient d'un YAML (campagne d'éval, recette
d'entraînement), donc d'une frontière externe. Elle nomme les scorers plutôt que
de les construire, de sorte qu'une config puisse être lue et validée sur une
machine qui ne saurait pas les instancier.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ScorerConfig(BaseModel):
    """Un scorer nommé et ses arguments de construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    """Clé du registre (`wer`, `dnsmos`, `tool_call`…)."""

    enabled: bool = True
    options: dict[str, Any] = Field(default_factory=dict)
    """kwargs passés au constructeur du scorer."""


class ScoringConfig(BaseModel):
    """Jeu de scorers d'une campagne."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scorers: tuple[ScorerConfig, ...] = ()

    asr_model_id: str = "openai/whisper-large-v3-turbo"
    """ASR de référence du WER — partagé par tous les scorers qui transcrivent."""

    asr_backend: Literal["transformers", "faster_whisper"] = "transformers"
    """Moteur ASR du WER.

    ``faster_whisper`` transcrit sur CPU en int8 : c'est ce qui rend le WER
    mesurable pendant un entraînement, là où le chemin ``transformers`` charge
    Whisper sur ``cuda:0`` et fait déborder la VRAM.
    """

    asr_model_size: str = "base"
    """Taille du modèle faster-whisper (``tiny``/``base``/``small``/…)."""

    asr_device: str | None = None
    """Où charger l'ASR. ``None`` = CUDA si disponible.

    À forcer sur ``"cpu"`` quand le GPU sert déjà : l'entraînement culmine à
    ~96 % de la VRAM d'une L4, et Whisper à côté la fait déborder.
    """

    judge_model_id: str = "gemini-3.6-flash"
    """Juge du scorer de raisonnement.

    ``gemini-2.0-flash`` a été retiré du service (404 « no longer available ») ;
    ce défaut écrasait celui de ``GeminiJudge`` et faisait échouer toute
    campagne pilotée par la config.
    """

    fail_on_unavailable: bool = False
    """Si vrai, un scorer indisponible fait échouer la campagne au lieu de la dégrader."""

    @property
    def enabled_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.scorers if s.enabled)

    @classmethod
    def with_defaults(cls) -> ScoringConfig:
        """Jeu complet : audio (wer, dnsmos, nisqa) + texte (tool_call, reasoning)."""
        return cls(
            scorers=(
                ScorerConfig(name="wer"),
                ScorerConfig(name="dnsmos"),
                ScorerConfig(name="nisqa"),
                ScorerConfig(name="tool_call"),
                ScorerConfig(name="reasoning"),
            )
        )
