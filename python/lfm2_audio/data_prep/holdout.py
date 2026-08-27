"""Keep benchmark material out of training, across EVERY source.

The FR corpora overlap: the user's own datasets and the student's both draw on
Common Voice FR. So carving a benchmark out of one source and excluding it
*from that source* is not enough — the same clip, and the same speaker, can
walk back in through another source under a different id. A contaminated gate
reports progress that is really memorisation, and nothing downstream can
detect it.

Each ASR benchmark therefore ships two exclusion lists next to its JSONL
(written by ``lfm2-asr-bench``):

* ``speakers.txt`` — every speaker the benchmark selected. Speaker-level, not
  clip-level, because a speaker's other recordings teach the model that voice.
* ``source_ids.txt`` — the original clip ids, for sources that carry them.

:class:`HoldoutFilter` loads those lists from several benchmarks at once and is
applied to every source at mix time.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

logger = logging.getLogger(__name__)

SPEAKERS_FILE = "speakers.txt"
SOURCE_IDS_FILE = "source_ids.txt"


def normalise_transcript(text: str) -> str:
    """A transcript reduced to a comparable key.

    Catches the same utterance re-encoded by another pipeline: accents folded,
    case and punctuation dropped, whitespace collapsed. Deliberately lossy —
    two different clips of the same sentence collide, and for a hold-out that
    is the safe direction to err in.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    kept = "".join(char if char.isalnum() or char.isspace() else " " for char in stripped)
    return " ".join(kept.split())


@dataclass
class HoldoutStats:
    """Why rows were dropped — the number that makes contamination visible."""

    kept: int = 0
    by_speaker: int = 0
    by_source_id: int = 0
    by_transcript: int = 0

    @property
    def dropped(self) -> int:
        return self.by_speaker + self.by_source_id + self.by_transcript

    def summary(self) -> str:
        return (
            f"{self.kept} gardés, {self.dropped} exclus "
            f"(locuteur {self.by_speaker}, id source {self.by_source_id}, transcript {self.by_transcript})"
        )


@dataclass
class HoldoutFilter:
    """Rejects any training row that belongs to an evaluation benchmark."""

    speakers: set[str] = field(default_factory=set)
    source_ids: set[str] = field(default_factory=set)
    transcripts: set[str] = field(default_factory=set)
    stats: HoldoutStats = field(default_factory=HoldoutStats)

    @classmethod
    def from_benchmarks(cls, directories: Iterable[Path], *, transcripts: Iterable[str] = ()) -> Self:
        """Load the exclusion lists shipped by each benchmark directory.

        A missing list is not an error: FLEURS has no speaker ids, and a
        benchmark built without ``--id-column`` has no source ids. What would
        be an error is silently loading nothing, so each directory logs what
        it contributed.
        """
        speakers: set[str] = set()
        source_ids: set[str] = set()
        for directory in directories:
            found_speakers = _read_lines(directory / SPEAKERS_FILE)
            found_ids = _read_lines(directory / SOURCE_IDS_FILE)
            speakers |= found_speakers
            source_ids |= found_ids
            logger.info(
                "hold-out %s : %d locuteurs, %d ids source",
                directory.name,
                len(found_speakers),
                len(found_ids),
            )
        return cls(
            speakers=speakers,
            source_ids=source_ids,
            transcripts={normalise_transcript(text) for text in transcripts if text.strip()},
        )

    @property
    def is_empty(self) -> bool:
        """True when nothing would ever be excluded — almost always a mistake."""
        return not (self.speakers or self.source_ids or self.transcripts)

    def excludes(self, row: Mapping[str, Any], *, speaker_key: str = "speaker", id_key: str = "id") -> bool:
        """Whether this training row overlaps the held-out benchmarks."""
        speaker = str(row.get(speaker_key, "") or "")
        if speaker and speaker in self.speakers:
            self.stats.by_speaker += 1
            return True
        source_id = str(row.get(id_key, "") or "")
        if source_id and source_id in self.source_ids:
            self.stats.by_source_id += 1
            return True
        if self.transcripts:
            text = normalise_transcript(str(row.get("text", "") or ""))
            if text and text in self.transcripts:
                self.stats.by_transcript += 1
                return True
        self.stats.kept += 1
        return False

    def keep(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        speaker_key: str = "speaker",
        id_key: str = "id",
    ) -> list[Mapping[str, Any]]:
        """The rows safe to train on."""
        return [row for row in rows if not self.excludes(row, speaker_key=speaker_key, id_key=id_key)]


def _read_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
