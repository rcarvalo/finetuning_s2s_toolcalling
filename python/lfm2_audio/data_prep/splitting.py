"""Carve a held-out evaluation split that a comparison can actually rely on.

Weekend step 4b. The shipped test split holds twelve cases: enough to prove the
vanilla model never calls a tool, far too few to tell a real gain from noise
once fine-tuning starts.

Growing it is not a matter of sampling at random. The metric aggregates over
three very different behaviours — call `web_search`, call `db_query`, or abstain
— so a split that drifts from the source distribution moves the score on its
own. :func:`stratified_split` keeps each target's share, which makes two
campaigns comparable even when the split is regenerated.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


def target_of(dialogue: dict[str, Any]) -> str:
    """The behaviour a dialogue exercises: a tool name, or ``"none"``.

    Read from the assistant turn rather than ``meta``: the turn is what the
    model is trained and scored on, and generation metadata can be stale.
    """
    for turn in dialogue.get("turns", []):
        if turn.get("role") == "assistant" and turn.get("tool_calls"):
            calls = turn["tool_calls"]
            if calls:
                return str(calls[0].get("name") or "unknown")
    return "none"


@dataclass(frozen=True, slots=True)
class SplitReport:
    """Sizes and per-target composition of both sides."""

    train: int
    test: int
    train_targets: dict[str, int]
    test_targets: dict[str, int]

    def summary(self) -> str:
        shares = ", ".join(
            f"{target}={count} ({100 * count / max(self.test, 1):.0f}%)"
            for target, count in sorted(self.test_targets.items())
        )
        return f"train {self.train} / test {self.test} — test composition: {shares}"


def stratified_split(
    dialogues: Sequence[dict[str, Any]],
    *,
    test_size: int,
    seed: int = 0,
    target: Callable[[dict[str, Any]], str] = target_of,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], SplitReport]:
    """Split off ``test_size`` dialogues, preserving the target distribution.

    ``target`` names the behaviour to balance; the default reads the assistant
    turn, and flat Hub rows can pass their own reader.

    Deterministic for a given ``seed`` so a regenerated split is the same split.
    Raises when the corpus is too small to honour the request, rather than
    silently returning a smaller — and incomparable — test set.
    """
    if test_size <= 0:
        message = f"test_size must be > 0, got {test_size}"
        raise ValueError(message)
    if test_size >= len(dialogues):
        message = f"test_size {test_size} needs fewer than the {len(dialogues)} available dialogues"
        raise ValueError(message)

    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dialogue in dialogues:
        by_target[target(dialogue)].append(dialogue)

    rng = random.Random(seed)
    test: list[dict[str, Any]] = []
    # Largest groups first: rounding leftovers land where they distort least.
    for name in sorted(by_target, key=lambda t: -len(by_target[t])):
        group = by_target[name]
        rng.shuffle(group)
        quota = round(test_size * len(group) / len(dialogues))
        test.extend(group[:quota])

    # Rounding can miss the target by a unit or two; top up from the largest pool.
    if len(test) < test_size:
        chosen = {id(d) for d in test}
        spare = [d for d in dialogues if id(d) not in chosen]
        rng.shuffle(spare)
        test.extend(spare[: test_size - len(test)])
    test = test[:test_size]

    in_test = {id(d) for d in test}
    train = [d for d in dialogues if id(d) not in in_test]

    return (
        train,
        test,
        SplitReport(
            train=len(train),
            test=len(test),
            train_targets=_count(train, target),
            test_targets=_count(test, target),
        ),
    )


def _count(dialogues: Sequence[dict[str, Any]], target: Callable[[dict[str, Any]], str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for dialogue in dialogues:
        counts[target(dialogue)] += 1
    return dict(counts)
