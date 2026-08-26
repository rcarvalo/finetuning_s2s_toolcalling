"""Our ``ScoreResult`` values, in the shape Inspect's viewer expects.

Deliberately a translation and not a reimplementation: the same
:class:`~lfm2_audio.scorer.base.BaseScorer` grades a training step, a campaign
report and this log, so a number shown in the viewer is the number the report
carries. Re-deriving anything here would let the two drift.

``details`` becomes ``Score.metadata``, which is what makes the failure anatomy
readable in the Scoring tab: expected call, predicted call, the offending
argument and its similarity.
"""

from __future__ import annotations

from inspect_ai.scorer import Score

from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.status import ScoreStatus

UNMEASURED_EXPLANATION = {
    ScoreStatus.UNAVAILABLE: "metric unavailable on the machine that ran this campaign",
    ScoreStatus.SKIPPED: "not applicable to this sample",
    ScoreStatus.FAILED: "scorer raised while measuring",
}


def to_inspect_score(result: ScoreResult) -> Score | None:
    """One measured result → one Inspect score. ``None`` when nothing was measured.

    A skipped or unavailable metric is dropped rather than reported as zero:
    the distinction between "no measurement" and "a bad measurement" is the one
    thing an eval must never blur.
    """
    if result.status is not ScoreStatus.OK or result.value is None:
        return None
    return Score(
        value=result.value,
        explanation=result.reason or f"{result.scorer} on this sample",
        metadata=dict(result.details),
    )


def unmeasured_note(result: ScoreResult) -> str:
    """Human-readable reason a metric produced no score, for sample metadata."""
    prefix = UNMEASURED_EXPLANATION.get(result.status, "not measured")
    return f"{prefix}: {result.reason}" if result.reason else prefix
