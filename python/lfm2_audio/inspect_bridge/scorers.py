"""Our scorers, exposed to Inspect — through the evaluation toolkit's adapter.

The same :class:`~avet.scoring.base_scorer.BaseScorer` grades a training
step, a campaign report and an Inspect run. What this module adds is the
LFM2 text cleaner, so what is judged and what is shown as the spoken answer
cannot diverge. A metric that produced nothing becomes a *stub* (``nan`` +
``metadata.stub``) that every aggregate skips — never a zero, never a
sentinel.
"""

from __future__ import annotations

from typing import Any

from avet.bridge.eval_sample_adapter import EvalSampleAdapter
from avet.bridge.inspect_scorer_factory import InspectScorerFactory
from avet.bridge.scorer_adapter import ScorerAdapter
from avet.scoring.base_scorer import BaseScorer
from avet.scoring.eval_sample import EvalSample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import TaskState

from lfm2_audio.avet_components.text_cleaner import Lfm2TextCleaner
from lfm2_audio.ds.scoring_config import ScoringConfig

_ADAPTER = EvalSampleAdapter(Lfm2TextCleaner())


def to_eval_sample(state: TaskState) -> EvalSample:
    """Rebuild the sample our scorers expect from Inspect's turn state (LFM2 markers cleaned)."""
    return _ADAPTER.from_state(state)


def wrap(base: BaseScorer) -> Scorer:
    """One of our scorers as an Inspect scorer."""
    return ScorerAdapter(adapter=_ADAPTER).wrap(base)


def lfm2_scorer(
    name: str,
    *,
    scoring: ScoringConfig | None = None,
    **kwargs: Any,  # noqa: ANN401 — kwargs of the underlying scorer
) -> Scorer:
    """Build one of our registered scorers, by name, for an Inspect task.

    Built through the toolkit's factory so shared dependencies (WER's
    transcriber, the judge, the LFM2 parser) are injected the way every
    campaign gets them.
    """
    return InspectScorerFactory(scoring or ScoringConfig()).build_named(name, **kwargs)


__all__ = ["lfm2_scorer", "to_eval_sample", "wrap"]
