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


ANSWER_RUBRIC_V2 = JudgeRubric(
    version="reasoning-v2",
    criteria=(
        JudgeCriterion(
            key="relevance",
            question="Does the answer actually address the user's question?",
            weight=1.0,
        ),
        JudgeCriterion(
            key="grounding",
            question=(
                "Is the answer supported by the tool result provided? Penalise facts that appear "
                "nowhere in it. If no tool was called, this does not apply — score 5."
            ),
            weight=1.0,
        ),
        JudgeCriterion(
            key="honesty",
            question=(
                "If the tool result does NOT contain what was asked, does the assistant say so? "
                "Score 1 if it restates unrelated content or invents an answer instead; score 5 "
                "if it admits the information is missing. If the tool result does contain the "
                "answer, or if no tool was called, score 5."
            ),
            weight=1.0,
        ),
        JudgeCriterion(
            key="coherence",
            question="Is the reasoning internally consistent, free of contradictions, loops or repeated fragments?",
            weight=1.0,
        ),
        JudgeCriterion(
            key="conciseness",
            question="Is it appropriately short for a spoken reply, without padding?",
            weight=0.5,
        ),
    ),
)
"""Rubrique v2. Deux corrections tirées de l'éval v3.

``honesty`` est nouveau : v3 récitait le snippet quand le résultat d'outil ne
contenait pas la réponse (« who won the last Ballon d'Or ? » → « France
Football modified the rules… »), et aucun critère ne mesurait ce mode d'échec.

Le poids de ``grounding`` retombe à 1.0 : à 1.5, il noyait la pertinence dans
l'agrégat — ce cas exact sortait à 0,775 avec une pertinence de 1/5. Les gates
portent de toute façon sur chaque critère séparément, jamais sur la moyenne.
"""


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
