"""Métriques réutilisables — évaluation, entraînement, analyse ad hoc.

Une seule abstraction, :class:`BaseScorer`, sert les trois contextes : c'est ce
qui évite d'avoir un WER dans la pipeline d'éval et un autre dans la boucle
d'entraînement.

>>> from lfm2_audio.ds.scoring_config import ScoringConfig      # doctest: +SKIP
>>> from lfm2_audio.scorer import ScorerFactory                 # doctest: +SKIP
>>> scorers = ScorerFactory(ScoringConfig.with_defaults()).build_all()  # doctest: +SKIP
>>> [s.score(sample) for s in scorers]                          # doctest: +SKIP

Les scorers dont les dépendances manquent ne font pas échouer la campagne : ils
sont remplacés par un :class:`MissingScorer` qui rend ``UNAVAILABLE`` avec la
raison. Une éval partielle se lit donc comme telle dans le rapport.
"""

from lfm2_audio.scorer.aggregate import MetricSummary
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.factory import ScorerFactory
from lfm2_audio.scorer.missing import MissingScorer
from lfm2_audio.scorer.registry import SCORERS, ScorerRegistry, UnknownScorerError
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.spec import ScorerSpec
from lfm2_audio.scorer.status import ScoreStatus

__all__ = [
    "SCORERS",
    "BaseScorer",
    "EvalSample",
    "MetricSummary",
    "MissingScorer",
    "ScoreResult",
    "ScoreStatus",
    "ScorerFactory",
    "ScorerRegistry",
    "ScorerSpec",
    "UnknownScorerError",
]
