"""``JudgeRubric`` — critères de notation du juge LLM.

La rubrique est un objet, pas une chaîne enfouie dans le prompt : elle est
versionnable, comparable d'une campagne à l'autre, et surchargeable depuis une
config YAML. Deux évals menées avec des rubriques différentes ne sont pas
comparables — l'expliciter évite de le découvrir après coup.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SCALE = 5


@dataclass(frozen=True, slots=True)
class JudgeCriterion:
    """Un axe de notation."""

    key: str
    question: str
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class JudgeRubric:
    """Ensemble pondéré de critères, sur une échelle commune."""

    criteria: tuple[JudgeCriterion, ...]
    scale: int = DEFAULT_SCALE
    version: str = "v1"

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.criteria)

    def weighted_mean(self, scores: dict[str, float]) -> float:
        """Moyenne pondérée, normalisée entre 0 et 1.

        Les critères absents de la réponse du juge sont ignorés plutôt que
        comptés zéro : une note manquante n'est pas une mauvaise note.
        """
        graded = [(c, scores[c.key]) for c in self.criteria if c.key in scores]
        if not graded:
            return 0.0
        total_weight = sum(c.weight for c, _ in graded)
        weighted = sum(c.weight * value for c, value in graded)
        return weighted / (total_weight * self.scale)

    def as_prompt_block(self) -> str:
        """Critères rendus pour le prompt du juge."""
        lines = [f"- {c.key}: {c.question} (1-{self.scale})" for c in self.criteria]
        return "\n".join(lines)


REASONING_RUBRIC = JudgeRubric(
    version="reasoning-v1",
    criteria=(
        JudgeCriterion(
            key="relevance",
            question="Does the answer actually address the user's question?",
            weight=1.0,
        ),
        JudgeCriterion(
            key="grounding",
            question=("Is the answer supported by the tool result provided? Penalise facts that appear nowhere in it."),
            weight=1.5,
        ),
        JudgeCriterion(
            key="coherence",
            question="Is the reasoning internally consistent and free of contradictions?",
            weight=1.0,
        ),
        JudgeCriterion(
            key="conciseness",
            question="Is it appropriately short for a spoken reply, without padding?",
            weight=0.5,
        ),
    ),
)
"""Rubrique par défaut. L'ancrage pèse le plus : c'est le mode d'échec dominant
d'un assistant qui appelle un outil puis parle à côté du résultat."""
