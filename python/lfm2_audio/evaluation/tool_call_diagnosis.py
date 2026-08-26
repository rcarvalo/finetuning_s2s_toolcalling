"""``ToolCallDiagnosis`` — the anatomy of one tool-calling case, evidence kept.

``score_case`` computes everything needed to explain a failure and then throws
the evidence away: the similarity that fell below the threshold is computed one
line before being discarded, and a report ends up saying ``call: false`` and
nothing else. Diagnosing a run then means re-deriving by hand what the scorer
already knew.

This module computes the same verdicts and *keeps* them, so a case can answer
"why": which tool was expected, which was emitted, which argument diverged, by
how much, and what the raw span looked like. ``score_case`` and ``_args_match``
delegate here — one implementation, no drift between evaluation and training.

The single ``outcome`` label is what turns an aggregate ("tool_call 0.830") into
something actionable ("45 wrong_arguments, 3 spurious_call, 0 parse_error").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lfm2_audio.core.chat_format import TOOL_CALL_SPAN
from lfm2_audio.evaluation.argument_match import ArgMatch, ArgumentMismatch, diff_arguments
from lfm2_audio.orchestrator.tool_parser import StreamingToolCallParser

OUTCOMES: tuple[str, ...] = (
    "no_generation",
    "parse_error",
    "unterminated_call",
    "missing_call",
    "spurious_call",
    "wrong_tool",
    "arity_mismatch",
    "wrong_arguments",
    "correct_call",
    "correct_abstention",
)
"""Every case gets exactly one, evaluated in this precedence order."""


@dataclass(frozen=True, slots=True)
class ToolCallDiagnosis:
    """What a tool-calling case did, and why it counted as it did."""

    case_id: str
    outcome: str
    parsed: bool
    expected_call: bool
    predicted_call: bool
    name_correct: bool
    call_correct: bool
    expected: list[dict[str, Any]] = field(default_factory=list)
    predicted: list[dict[str, Any]] = field(default_factory=list)
    argument_mismatches: tuple[ArgumentMismatch, ...] = ()
    parse_errors: tuple[str, ...] = ()
    raw_span: str = ""
    arg_match: ArgMatch = "exact"
    threshold: float = 0.7

    @classmethod
    def of(
        cls,
        case_id: str,
        predicted_text: str,
        expected_calls: list[dict[str, Any]],
        *,
        arg_match: ArgMatch = "exact",
        threshold: float = 0.7,
    ) -> ToolCallDiagnosis:
        """Diagnose one case from the model's final text and the expected calls."""

        parser = StreamingToolCallParser()
        predicted = [{"name": c.name, "arguments": c.arguments} for c in parser.feed(predicted_text)]
        parse_errors = tuple(parser.errors)
        # A span opened and never closed leaves the parser mid-call with no
        # error recorded, so it used to read as "emitted no call" — a silent
        # miss on a positive case, a false success on a negative one. vLLM
        # strips the stop token and leaves spans open, so this is routine.
        unterminated = parser.in_tool_call
        expected_call = bool(expected_calls)
        predicted_call = bool(predicted) or bool(parse_errors) or unterminated

        name_correct, call_correct, mismatches = cls._match(
            predicted, expected_calls, arg_match=arg_match, threshold=threshold
        )
        span = TOOL_CALL_SPAN.search(predicted_text)
        return cls(
            case_id=case_id,
            outcome=cls._outcome(
                predicted_text=predicted_text,
                parse_errors=parse_errors,
                unterminated=unterminated,
                expected_call=expected_call,
                predicted_call=predicted_call,
                predicted=predicted,
                expected=expected_calls,
                call_correct=call_correct,
            ),
            parsed=not parse_errors,
            expected_call=expected_call,
            predicted_call=predicted_call,
            name_correct=name_correct,
            call_correct=call_correct,
            expected=list(expected_calls),
            predicted=predicted,
            argument_mismatches=mismatches,
            parse_errors=parse_errors,
            raw_span=span.group(0) if span else "",
            arg_match=arg_match,
            threshold=threshold,
        )

    @staticmethod
    def _match(
        predicted: list[dict[str, Any]],
        expected: list[dict[str, Any]],
        *,
        arg_match: ArgMatch,
        threshold: float,
    ) -> tuple[bool, bool, tuple[ArgumentMismatch, ...]]:
        """Unordered bipartite match; on failure, why the closest pairing failed."""

        if not (expected and predicted):
            return False, False, ()

        name_correct = sorted(str(c["name"]) for c in expected) == sorted(str(c["name"]) for c in predicted)
        remaining = list(expected)
        mismatches: list[ArgumentMismatch] = []
        matched = 0
        for index, call in enumerate(predicted):
            same_name = [e for e in remaining if e["name"] == call["name"]]
            paired = None
            for candidate in same_name:
                if not diff_arguments(
                    call.get("arguments", {}),
                    candidate.get("arguments", {}),
                    arg_match=arg_match,
                    threshold=threshold,
                    call_index=index,
                ):
                    paired = candidate
                    break
            if paired is not None:
                remaining.remove(paired)
                matched += 1
            elif same_name:
                # Same tool, arguments rejected: report against the first
                # candidate, which is the pairing a reader would assume.
                mismatches.extend(
                    diff_arguments(
                        call.get("arguments", {}),
                        same_name[0].get("arguments", {}),
                        arg_match=arg_match,
                        threshold=threshold,
                        call_index=index,
                    )
                )
        call_correct = matched == len(expected) == len(predicted)
        return name_correct, call_correct, tuple(mismatches)

    @staticmethod
    def _outcome(
        *,
        predicted_text: str,
        parse_errors: tuple[str, ...],
        unterminated: bool,
        expected_call: bool,
        predicted_call: bool,
        predicted: list[dict[str, Any]],
        expected: list[dict[str, Any]],
        call_correct: bool,
    ) -> str:
        """The single label for this case, in precedence order."""
        if not predicted_text.strip():
            return "no_generation"
        if parse_errors:
            return "parse_error"
        if unterminated:
            return "unterminated_call"
        if expected_call and not predicted_call:
            return "missing_call"
        if not expected_call:
            return "spurious_call" if predicted_call else "correct_abstention"
        # Which tools, then how many: two calls to the SAME tool make the name
        # *lists* differ while the tools chosen are right, and reading that as
        # "wrong tool" would send a reader looking for a routing bug.
        if {str(c["name"]) for c in predicted} != {str(c["name"]) for c in expected}:
            return "wrong_tool"
        if len(predicted) != len(expected):
            return "arity_mismatch"
        return "correct_call" if call_correct else "wrong_arguments"

    @property
    def succeeded(self) -> bool:
        """Turn success: the right call, or a correct refusal to call."""
        return self.call_correct if self.expected_call else not self.predicted_call

    def as_details(self) -> dict[str, Any]:
        """Payload for ``ScoreResult.details`` — the six historical booleans, plus the evidence."""
        return {
            "parse": self.parsed,
            "relevance": self.expected_call == self.predicted_call,
            "name": self.name_correct,
            "call": self.call_correct,
            "expected_call": self.expected_call,
            "predicted_call": self.predicted_call,
            "outcome": self.outcome,
            "expected": self.expected,
            "predicted": self.predicted,
            "argument_mismatches": [m.as_dict() for m in self.argument_mismatches],
            "parse_errors": list(self.parse_errors),
            "raw_span": self.raw_span,
            "arg_match": self.arg_match,
            "threshold": self.threshold,
        }
