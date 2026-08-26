"""Selection logic for ASR benchmark sets pulled from HF datasets.

The benchmark defines the held-out set, not the other way round: the speakers
selected here are recorded next to the JSONL, and the training mixer excludes
them later. Picking eval first is what makes "held-out" checkable instead of
declarative.

Pure logic — the download/IO half lives in ``cli/data/make_asr_bench.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AsrCandidate:
    """One dataset row, reduced to what selection needs."""

    sample_id: str
    transcript: str
    speaker: str = ""
    score: float | None = None


@dataclass
class AsrClipSelector:
    """Accepts clips until ``limit``, enforcing quality and speaker diversity.

    ``min_score`` drops low-quality clips (e.g. distillmos on the student
    dataset); ``max_per_speaker`` keeps one loud speaker from owning the
    benchmark. Order is first-come: the caller controls shuffling upstream.
    """

    limit: int
    min_score: float | None = None
    max_per_speaker: int | None = None
    _per_speaker: dict[str, int] = field(default_factory=dict, init=False)
    _accepted: int = field(default=0, init=False)

    def offer(self, candidate: AsrCandidate) -> bool:
        if self.full or not candidate.transcript.strip():
            return False
        if self.min_score is not None and (candidate.score is None or candidate.score < self.min_score):
            return False
        taken = self._per_speaker.get(candidate.speaker, 0)
        if self.max_per_speaker is not None and candidate.speaker and taken >= self.max_per_speaker:
            return False
        self._per_speaker[candidate.speaker] = taken + 1
        self._accepted += 1
        return True

    @property
    def full(self) -> bool:
        return self._accepted >= self.limit

    @property
    def accepted(self) -> int:
        return self._accepted

    @property
    def speakers(self) -> set[str]:
        return {speaker for speaker in self._per_speaker if speaker}


def asr_dialogue(sample_id: str, transcript: str, audio_relpath: str, lang: str) -> dict[str, Any]:
    """One benchmark case in the shared dialogue JSONL schema.

    The reference transcript is the assistant turn (``reference_answer`` once
    loaded as a :class:`Question`): the model hears the clip and must produce
    that text.
    """
    return {
        "id": sample_id,
        "tools": [],
        "meta": {"lang": lang, "task": "asr"},
        "turns": [
            {"role": "user", "audio": audio_relpath},
            {"role": "assistant", "text": transcript},
        ],
    }
