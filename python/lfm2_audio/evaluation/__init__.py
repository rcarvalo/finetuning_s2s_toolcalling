"""Évaluation : jeu de questions, génération, notation, rapport.

>>> from lfm2_audio.evaluation import EvaluationPipeline, QuestionSet   # doctest: +SKIP
>>> questions = QuestionSet.from_jsonl("benchmark/toolcalling_en/cases.sample.jsonl")  # doctest: +SKIP
>>> report = EvaluationPipeline(scorers).run(questions, generator)      # doctest: +SKIP
>>> print(report.render())                                              # doctest: +SKIP

Les métriques viennent de :mod:`lfm2_audio.scorer` — les mêmes objets que ceux
suivis pendant l'entraînement, pour que les deux chiffres soient comparables.
"""

from lfm2_audio.evaluation.generator import ResponseGenerator
from lfm2_audio.evaluation.pipeline import EvaluationPipeline
from lfm2_audio.evaluation.question import Question
from lfm2_audio.evaluation.question_set import QuestionSet
from lfm2_audio.evaluation.report import EvaluationReport, SampleReport

__all__ = [
    "EvaluationPipeline",
    "EvaluationReport",
    "Question",
    "QuestionSet",
    "ResponseGenerator",
    "SampleReport",
]
