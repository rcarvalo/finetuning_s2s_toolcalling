"""Listening bench: generate answers, judge them by ear, keep the verdicts.

Automatic metrics can be wrong for weeks without anything flagging it — the EN
baseline scored every clip at 1.6 because a calibration polynomial saturated,
and only a listener noticed. Stored human verdicts turn that from a lucky catch
into a check: correlate them with WER and DNSMOS, and a metric that stops
tracking perception becomes visible.
"""

from lfm2_audio.bench.rating import AXES, SCALE_MAX, SCALE_MIN, Rating
from lfm2_audio.bench.session import BenchSession
from lfm2_audio.bench.source import AnswerSource
from lfm2_audio.bench.store import RatingStore

__all__ = [
    "AXES",
    "SCALE_MAX",
    "SCALE_MIN",
    "AnswerSource",
    "BenchSession",
    "Rating",
    "RatingStore",
]
