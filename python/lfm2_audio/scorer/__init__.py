"""Reusable metrics — evaluation, training, ad hoc analysis.

They now live in the evaluation toolkit (``avet``); these modules keep the
historical import paths so callers and configs do not move.
"""

from avet.errors import UnknownScorerError
from avet.scoring.base_scorer import BaseScorer
from avet.scoring.eval_sample import EvalSample
from avet.scoring.metric_summary import MetricSummary
from avet.scoring.missing_scorer import MissingScorer
from avet.scoring.score_result import ScoreResult
from avet.scoring.score_status import ScoreStatus
from avet.scoring.scorer_factory import ScorerFactory
from avet.scoring.scorer_registry import SCORERS, ScorerRegistry
from avet.scoring.scorer_spec import ScorerSpec

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
