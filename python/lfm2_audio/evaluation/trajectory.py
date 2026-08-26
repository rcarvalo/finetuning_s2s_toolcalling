"""``Trajectory`` — what happened during one turn, step by step.

A score says a turn failed; the trajectory says where. The orchestrator already
emits the steps (call, tool result, spoken answer, timings) and then discards
everything but the final text, which is why diagnosing a run means guessing
backwards from an aggregate.

Ordered, append-only, JSON-serialisable: it is written into the sample archive
next to the audio, so a run can be replayed, exported to the Inspect viewer, or
re-scored months later without the machine that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

StepKind = Literal["prompt", "tool_call", "tool_result", "answer", "error"]

STEP_KINDS: tuple[StepKind, ...] = ("prompt", "tool_call", "tool_result", "answer", "error")


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One step of a turn: what happened, and how long it took to get there.

    ``elapsed_s`` is measured from the start of the turn, not from the previous
    step: that is what makes a late first token visible when reading the list
    top to bottom.
    """

    kind: StepKind
    content: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        step: dict[str, Any] = {"kind": self.kind, "content": self.content}
        if self.payload:
            step["payload"] = self.payload
        if self.elapsed_s is not None:
            step["elapsed_s"] = round(self.elapsed_s, 3)
        return step

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrajectoryStep:
        kind = payload.get("kind", "answer")
        return cls(
            kind=kind if kind in STEP_KINDS else "answer",
            content=str(payload.get("content", "")),
            payload=dict(payload.get("payload", {})),
            elapsed_s=payload.get("elapsed_s"),
        )


@dataclass(slots=True)
class Trajectory:
    """The ordered steps of one turn."""

    steps: list[TrajectoryStep] = field(default_factory=list)

    def add(self, kind: StepKind, content: str = "", **payload: Any) -> TrajectoryStep:  # noqa: ANN401
        """Append a step; ``elapsed_s`` is passed as a payload key when known."""
        elapsed = payload.pop("elapsed_s", None)
        step = TrajectoryStep(kind=kind, content=content, payload=payload, elapsed_s=elapsed)
        self.steps.append(step)
        return step

    def of_kind(self, kind: StepKind) -> tuple[TrajectoryStep, ...]:
        return tuple(step for step in self.steps if step.kind == kind)

    @property
    def is_empty(self) -> bool:
        return not self.steps

    @property
    def failed(self) -> bool:
        return any(step.kind == "error" for step in self.steps)

    def as_list(self) -> list[dict[str, Any]]:
        return [step.as_dict() for step in self.steps]

    @classmethod
    def from_list(cls, payload: list[dict[str, Any]] | None) -> Trajectory:
        return cls(steps=[TrajectoryStep.from_dict(step) for step in payload or []])
