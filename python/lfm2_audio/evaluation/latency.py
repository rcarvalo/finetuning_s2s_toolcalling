"""Mesure de latence du serving : TTFA, RTF, débit.

Le **TTFA** (time-to-first-audio) est la métrique qui compte : c'est le délai
avant que l'utilisateur entende quelque chose, donc le seul chiffre comparable
aux services commerciaux. Le total et le RTF disent si la lecture tiendra sans
trous une fois le premier chunk parti.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

from lfm2_audio.evaluation.latency_sample import LatencySample

logger = logging.getLogger(__name__)

TTFA_TARGET_S = 0.5
"""Cible haute de l'objectif « 200-500 ms »."""

DEFAULT_PROMPTS = (
    "Hello, who are you?",
    "What time is it in Paris?",
    "Tell me a short story.",
)

# TTFA is prompt-length-sensitive, so a FR latency figure quoted off EN prompts
# would not be a FR figure. Same lengths and same shapes as the EN set, so the
# two series stay comparable.
DEFAULT_PROMPTS_FR = (
    "Bonjour, qui es-tu ?",
    "Quelle heure est-il à Paris ?",
    "Raconte-moi une petite histoire.",
)

PROMPTS_BY_LANGUAGE = {"en": DEFAULT_PROMPTS, "fr": DEFAULT_PROMPTS_FR}


@dataclass(frozen=True, slots=True)
class LatencyReport:
    """Agrégat des mesures d'une campagne."""

    samples: tuple[LatencySample, ...] = field(default_factory=tuple)

    @property
    def measured(self) -> list[float]:
        """TTFA effectivement observés (les tours muets n'en ont pas)."""
        return [s.ttfa_s for s in self.samples if s.ttfa_s is not None]

    @property
    def ttfa_p50(self) -> float | None:
        values = self.measured
        return statistics.median(values) if values else None

    @property
    def ttfa_p95(self) -> float | None:
        values = self.measured
        if not values:
            return None
        if len(values) < 20:
            return max(values)
        return statistics.quantiles(values, n=20)[18]

    @property
    def median_rtf(self) -> float | None:
        factors = [s.real_time_factor for s in self.samples if s.real_time_factor is not None]
        return statistics.median(factors) if factors else None

    @property
    def meets_target(self) -> bool:
        return self.ttfa_p50 is not None and self.ttfa_p50 <= TTFA_TARGET_S

    def diagnose(self) -> str | None:
        """Message d'aide quand aucun TTFA n'a pu être mesuré.

        Deux pannes très différentes se ressemblent en sortie ; les distinguer
        ici épargne des heures de recherche au mauvais endroit.
        """
        if self.measured:
            return None
        if all(sample.audio_frames == 0 for sample in self.samples):
            return (
                "le stage 0 n'émet AUCUNE frame audio (pas de <|text_end|> dans la "
                "génération) → prompt ou modèle, pas la plomberie. Vérifier --system "
                "et le checkpoint."
            )
        return (
            "des frames audio sont générées par le stage 0 mais aucun chunk n'arrive "
            "du stage 1 → plomberie connector/stage 1."
        )


def format_ms(seconds: float | None) -> str:
    """Secondes → millisecondes lisibles, ou tiret si non mesuré."""
    return f"{seconds * 1000:.0f} ms" if seconds is not None else "—"
