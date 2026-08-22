"""Merge dialogue sources into one clean training corpus.

Weekend step 4. Two failure modes make a tool-calling corpus worthless, and
neither raises an error on its own:

* **duplicates** â the same utterance repeated inflates the apparent size and
  biases the model toward one phrasing;
* **train/test leakage** â an utterance present in both splits turns the
  evaluation into a memorization check, and the step-7 comparison becomes
  meaningless.

:func:`curate` removes both, and reports what it dropped: a silent filter is
indistinguishable from a filter that never ran.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_APOSTROPHES = re.compile(r"['’ʼ]")
_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_utterance(text: str) -> str:
    """Fold an utterance to its comparison key: case, accents and punctuation out.

    Two spoken prompts that differ only by a comma are the same example for a
    deduplication purpose, and the same leak for a contamination purpose.

    Apostrophes are deleted rather than turned into a space, so a contraction
    matches its spelled-out form ("what's" and "whats" are one utterance).
    """
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = _APOSTROPHES.sub("", folded)
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", folded)).strip()


def first_user_text(dialogue: dict[str, Any]) -> str:
    """The user prompt a dialogue is keyed on (empty when there is none)."""
    for turn in dialogue.get("turns", []):
        if turn.get("role") == "user":
            return str(turn.get("text") or "")
    return ""


@dataclass(frozen=True, slots=True)
class CurationReport:
    """What the merge kept, and everything it threw away."""

    kept: int = 0
    duplicates: int = 0
    leaked: int = 0
    empty: int = 0
    per_source: dict[str, int] = field(default_factory=dict)

    @property
    def dropped(self) -> int:
        return self.duplicates + self.leaked + self.empty

    def summary(self) -> str:
        sources = ", ".join(f"{name}={count}" for name, count in self.per_source.items())
        return (
            f"kept {self.kept} dialogue(s) [{sources}] â dropped {self.dropped} "
            f"(duplicates {self.duplicates}, leaked {self.leaked}, empty {self.empty})"
        )


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def curate(
    sources: Mapping[str, Iterable[dict[str, Any]]],
    *,
    held_out: Iterable[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], CurationReport]:
    """Merge sources in order, dropping duplicates and anything in ``held_out``.

    Sources are merged in iteration order, so the first occurrence of an
    utterance wins â put the most trusted source first.
    """
    forbidden = {normalize_utterance(first_user_text(d)) for d in held_out}
    forbidden.discard("")

    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicates = leaked = empty = 0
    per_source: dict[str, int] = {}

    for name, dialogues in sources.items():
        per_source.setdefault(name, 0)
        for dialogue in dialogues:
            key = normalize_utterance(first_user_text(dialogue))
            if not key:
                empty += 1
            elif key in forbidden:
                leaked += 1
            elif key in seen:
                duplicates += 1
            else:
                seen.add(key)
                kept.append(dialogue)
                per_source[name] += 1

    return kept, CurationReport(
        kept=len(kept), duplicates=duplicates, leaked=leaked, empty=empty, per_source=per_source
    )


def write_jsonl(dialogues: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write dialogues back as JSONL, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for dialogue in dialogues:
            handle.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
    return target
