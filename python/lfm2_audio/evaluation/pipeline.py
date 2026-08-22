"""``EvaluationPipeline`` — génère les réponses puis les note.

Deux phases séparées volontairement : générer coûte du GPU, noter coûte de l'ASR
et des appels au juge. Les découpler permet de renoter un jeu de réponses déjà
produit — changer de rubrique ou ajouter une métrique sans refaire tourner le
modèle.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from lfm2_audio.evaluation.generator import ResponseGenerator
from lfm2_audio.evaluation.question_set import QuestionSet
from lfm2_audio.evaluation.report import EvaluationReport, SampleReport
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.sample import EvalSample

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Fait répondre un modèle à un jeu de questions, puis applique les scorers."""

    def __init__(self, scorers: Sequence[BaseScorer]) -> None:
        self._scorers = list(scorers)

    @property
    def scorers(self) -> tuple[BaseScorer, ...]:
        return tuple(self._scorers)

    def run(
        self,
        questions: QuestionSet,
        generator: ResponseGenerator,
        *,
        context: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Campagne complète : génération puis notation."""
        samples = list(self.generate(questions, generator))
        return self.score(samples, context={**(context or {}), **self._context(questions)})

    def generate(self, questions: QuestionSet, generator: ResponseGenerator) -> Iterable[EvalSample]:
        """Phase 1 : produire les réponses. Une question en échec n'arrête pas la campagne."""
        for index, question in enumerate(questions, start=1):
            logger.info("[%d/%d] %s", index, len(questions), question.question_id)
            try:
                yield generator.generate(question)
            except Exception as error:  # une question qui plante ne perd pas les autres
                logger.warning("génération échouée sur %s : %s", question.question_id, error)
                yield EvalSample(
                    sample_id=question.question_id,
                    prompt_text=question.text,
                    expected_calls=question.expected_calls,
                    metadata={"generation_error": f"{type(error).__name__}: {error}"},
                )

    def score(self, samples: Sequence[EvalSample], *, context: dict[str, Any] | None = None) -> EvaluationReport:
        """Phase 2 : noter des réponses déjà produites."""
        reports = [
            SampleReport(
                sample_id=sample.sample_id,
                results=tuple(scorer.score(sample) for scorer in self._scorers),
                predicted_text=sample.predicted_text,
            )
            for sample in samples
        ]
        return EvaluationReport.from_samples(reports, context=context)

    def _context(self, questions: QuestionSet) -> dict[str, Any]:
        return {
            "question_set": questions.source,
            "cases": len(questions),
            "positives": questions.positives,
            "scorers": [s.name for s in self._scorers],
        }
