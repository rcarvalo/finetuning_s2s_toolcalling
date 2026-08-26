"""``ScorerRegistry`` — catalogue des métriques disponibles.

Le registre sait ce qui existe et ce qui est installable **sans rien importer** :
c'est ce qui permet à `lfm2-eval --list-scorers` de tourner sur un portable, et à
une config d'entraînement de nommer un scorer que la machine d'entraînement
seule saura construire.
"""

from __future__ import annotations

import logging

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.scorer.spec import ScorerSpec

logger = logging.getLogger(__name__)


class UnknownScorerError(Lfm2AudioError):
    """Nom de scorer absent du registre."""


class ScorerRegistry:
    """Catalogue nom → :class:`ScorerSpec`."""

    def __init__(self, specs: tuple[ScorerSpec, ...] = ()) -> None:
        self._specs: dict[str, ScorerSpec] = {spec.name: spec for spec in specs}

    def register(self, spec: ScorerSpec) -> None:
        self._specs[spec.name] = spec

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def available(self) -> tuple[str, ...]:
        """Scorers dont les dépendances sont installées."""
        return tuple(name for name, spec in self._specs.items() if spec.is_installable)

    def describe(self, name: str) -> ScorerSpec:
        spec = self._specs.get(name)
        if spec is None:
            message = f"scorer inconnu : {name!r} (connus : {', '.join(self.names)})"
            raise UnknownScorerError(message)
        return spec

    def specs(self) -> tuple[ScorerSpec, ...]:
        return tuple(self._specs.values())


SCORERS = ScorerRegistry(
    (
        ScorerSpec(
            name="wer",
            module="lfm2_audio.scorer.audio.wer",
            class_name="WerScorer",
            requires=("torch", "transformers"),
            description="taux d'erreur mot de l'audio généré, re-transcrit",
        ),
        ScorerSpec(
            name="dnsmos",
            module="lfm2_audio.scorer.audio.dnsmos",
            class_name="DnsmosScorer",
            requires=("onnxruntime",),
            description="MOS P.835 prédit (sig/bak/ovrl), sans référence",
        ),
        ScorerSpec(
            name="utmos",
            module="lfm2_audio.scorer.audio.utmos",
            class_name="UtmosScorer",
            requires=("torch",),
            description="MOS de naturalité prédit (UTMOS), sans référence",
        ),
        ScorerSpec(
            name="nisqa",
            module="lfm2_audio.scorer.audio.nisqa",
            class_name="NisqaScorer",
            requires=("torch",),
            description="MOS NISQA v2 prédit, sans référence",
        ),
        ScorerSpec(
            name="tool_call",
            module="lfm2_audio.scorer.text.tool_call",
            class_name="ToolCallScorer",
            description="exactitude des tool calls (parse/relevance/name/call)",
        ),
        ScorerSpec(
            name="reasoning",
            module="lfm2_audio.scorer.text.reasoning",
            class_name="ReasoningScorer",
            requires=("google.genai",),
            description="qualité de la réponse jugée par un LLM",
        ),
    )
)
