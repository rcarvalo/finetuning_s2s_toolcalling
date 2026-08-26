"""Our scorers, exposed to Inspect — wrapped, never reimplemented.

The same :class:`~lfm2_audio.scorer.base.BaseScorer` grades a training step, a
campaign report and an Inspect run. Rewriting a metric against Inspect's API
would let the two drift, and the first symptom would be a number in the viewer
that no report can reproduce.

The adapter's only real job is turning what Inspect gives a scorer — the sample
and the model output — back into the :class:`EvalSample` our scorers read.
"""

from __future__ import annotations

import logging
from typing import Any

from inspect_ai.model import ContentAudio
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from lfm2_audio.inspect_bridge.audio import data_uri_to_waveform
from lfm2_audio.inspect_bridge.scores import to_inspect_score
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.registry import SCORERS
from lfm2_audio.scorer.sample import EvalSample

logger = logging.getLogger(__name__)

UNMEASURED = -1.0
"""Sentinel for a metric that produced nothing.

Inspect needs a value to aggregate; ``None`` would be counted as zero, which
would read as "the model scored badly" instead of "nothing was measured". The
explanation carries the reason, and the sentinel is out of every metric's range.
"""


def to_eval_sample(state: TaskState) -> EvalSample:
    """Rebuild the sample our scorers expect from Inspect's turn state."""
    completion = state.output.completion if state.output else ""
    audio = None
    message = state.output.message if state.output else None
    if message is not None and not isinstance(message.content, str):
        parts = [p for p in message.content if isinstance(p, ContentAudio)]
        audio = data_uri_to_waveform(parts[0].audio) if parts else None

    metadata: dict[str, Any] = dict(state.metadata or {})
    return EvalSample(
        sample_id=str(state.sample_id),
        prompt_text=str(state.input_text or ""),
        predicted_text=completion,
        predicted_audio=audio,
        reference_text=str(state.target) if state.target else "",
        expected_calls=metadata.get("expected_calls", []),
        tool_results=metadata.get("tool_results", []),
        metadata=metadata,
    )


def wrap(base: BaseScorer) -> Scorer:
    """One of our scorers as an Inspect scorer."""

    async def score(state: TaskState, target: Target) -> Score:
        result = base.score(to_eval_sample(state))
        translated = to_inspect_score(result)
        if translated is not None:
            return translated
        # Not measured: say so, rather than let a zero pass for a verdict.
        return Score(value=UNMEASURED, explanation=f"{result.status}: {result.reason}")

    return score


def lfm2_scorer(name: str, **kwargs: Any) -> Scorer:  # noqa: ANN401 — kwargs du scorer sous-jacent
    """Build one of our registered scorers, by name, for an Inspect task.

    Goes through the registry so a missing optional dependency degrades the way
    it does everywhere else — reported, never silently absent.
    """
    spec = SCORERS.describe(name)
    reason = spec.unavailable_reason()
    if reason:
        logger.warning("scorer %s indisponible : %s", name, reason)

    @scorer(metrics=[mean()], name=name)
    def build() -> Scorer:
        return wrap(spec.load()(**kwargs))

    return build()
