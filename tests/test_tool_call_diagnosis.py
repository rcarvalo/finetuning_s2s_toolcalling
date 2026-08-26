"""Tests for ``evaluation.tool_call_diagnosis.ToolCallDiagnosis``."""

from __future__ import annotations

import pytest

from lfm2_audio.core.chat_format import TOOL_CALL_END, TOOL_CALL_START
from lfm2_audio.evaluation.tool_call_diagnosis import OUTCOMES, ToolCallDiagnosis

WEATHER = [{"name": "web_search", "arguments": {"query": "current weather in Tokyo"}}]


def _span(inner: str) -> str:
    return f"{TOOL_CALL_START}{inner}{TOOL_CALL_END}"


def _diagnose(text: str, expected: list[dict] | None = None, **kwargs: object) -> ToolCallDiagnosis:
    return ToolCallDiagnosis.of("case", text, expected if expected is not None else WEATHER, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Un label par cas, dans l'ordre de précédence
# --------------------------------------------------------------------------- #


def test_should_label_a_matching_call_correct() -> None:
    result = _diagnose(_span('[web_search(query="current weather in Tokyo")]'))

    assert result.outcome == "correct_call"
    assert result.succeeded


def test_should_label_a_correct_abstention_when_no_call_was_expected() -> None:
    result = _diagnose("Hello! How can I help?", expected=[])

    assert result.outcome == "correct_abstention"
    assert result.succeeded


def test_should_label_a_call_on_a_negative_case_spurious() -> None:
    result = _diagnose(_span('[web_search(query="hello")]'), expected=[])

    assert result.outcome == "spurious_call"
    assert not result.succeeded


def test_should_label_a_missing_call_when_one_was_expected() -> None:
    result = _diagnose("Sure, let me help you with that.")

    assert result.outcome == "missing_call"
    assert not result.succeeded


def test_should_label_the_wrong_tool() -> None:
    result = _diagnose(_span('[db_query(query="current weather in Tokyo")]'))

    assert result.outcome == "wrong_tool"
    assert not result.name_correct


def test_should_label_a_diverging_argument() -> None:
    result = _diagnose(
        _span('[web_search(query="best pizza in Rome")]'),
        arg_match="token_f1",
        threshold=0.7,
    )

    assert result.outcome == "wrong_arguments"
    assert result.name_correct


def test_should_label_an_unparsable_span_a_parse_error() -> None:
    result = _diagnose(_span("[this is not a call!!"))

    assert result.outcome == "parse_error"
    assert not result.parsed
    assert result.parse_errors


def test_should_label_an_empty_generation() -> None:
    result = _diagnose("   ")

    assert result.outcome == "no_generation"


def test_should_label_an_extra_call_an_arity_mismatch() -> None:
    result = _diagnose(_span('[web_search(query="current weather in Tokyo"), web_search(query="tokyo")]'))

    assert result.outcome == "arity_mismatch"


def test_every_outcome_is_declared() -> None:
    """The viewer groups by outcome; an undeclared label would silently vanish."""
    produced = {
        _diagnose(_span('[web_search(query="current weather in Tokyo")]')).outcome,
        _diagnose("hello", expected=[]).outcome,
        _diagnose(_span('[web_search(query="x")]'), expected=[]).outcome,
        _diagnose("no call here").outcome,
        _diagnose(_span('[db_query(query="x")]')).outcome,
        _diagnose(_span("[bad!!")).outcome,
        _diagnose("").outcome,
    }

    assert produced <= set(OUTCOMES)


# --------------------------------------------------------------------------- #
# Les deux défauts que le diagnostic met au jour
# --------------------------------------------------------------------------- #


def test_should_not_count_an_unterminated_call_as_an_abstention() -> None:
    """A span with no end marker used to read as "emitted nothing", so on a
    negative case it scored as a correct abstention — a false success. vLLM
    strips the stop token and leaves spans open, so this is routine."""
    text = f'{TOOL_CALL_START}[web_search(query="current weather in Tokyo")]'

    result = _diagnose(text, expected=[])

    assert result.outcome == "unterminated_call"
    assert result.predicted_call
    assert not result.succeeded


def test_should_name_a_positional_argument_as_such() -> None:
    """`parse_tool_call_block` renames positional args `_positional_0`, which can
    never match an expected key: the cause is a format violation, not a wrong value."""
    result = _diagnose(_span('[web_search("current weather in Tokyo")]'))

    reasons = {m.reason for m in result.argument_mismatches}
    assert "positional_argument" in reasons


# --------------------------------------------------------------------------- #
# L'évidence conservée
# --------------------------------------------------------------------------- #


def test_should_keep_the_similarity_that_failed_the_threshold() -> None:
    result = _diagnose(
        _span('[web_search(query="weather Tokyo now please tell me")]'),
        arg_match="token_f1",
        threshold=0.9,
    )

    mismatch = result.argument_mismatches[0]
    assert mismatch.key == "query"
    assert mismatch.reason == "below_threshold"
    assert mismatch.similarity is not None
    assert 0.0 < mismatch.similarity < 0.9
    assert mismatch.threshold == 0.9


def test_should_keep_the_raw_span_and_both_sides_of_the_comparison() -> None:
    text = _span('[web_search(query="pizza")]')

    result = _diagnose(text)

    assert result.raw_span == text
    assert result.expected == WEATHER
    assert result.predicted == [{"name": "web_search", "arguments": {"query": "pizza"}}]


def test_should_report_a_missing_key() -> None:
    result = _diagnose(_span("[web_search()]"))

    assert [(m.key, m.reason) for m in result.argument_mismatches] == [("query", "missing_key")]


def test_details_should_keep_the_six_historical_booleans() -> None:
    """Existing readers of `details` must not break when the payload grows."""
    details = _diagnose(_span('[web_search(query="current weather in Tokyo")]')).as_details()

    assert {"parse", "relevance", "name", "call", "expected_call", "predicted_call"} <= set(details)
    assert details["outcome"] == "correct_call"


@pytest.mark.parametrize("arg_match", ["exact", "token_f1"])
def test_should_match_an_exact_call_under_every_arg_match(arg_match: str) -> None:
    result = _diagnose(_span('[web_search(query="current weather in Tokyo")]'), arg_match=arg_match)

    assert result.outcome == "correct_call"
