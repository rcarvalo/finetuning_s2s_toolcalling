"""``ToolCallScorer()`` keeps grading LFM2 spans without being handed a parser."""

from __future__ import annotations

from lfm2_audio.core.chat_format import render_tool_calls
from lfm2_audio.evaluation.tool_call_diagnosis import diagnose
from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.text.tool_call import ToolCallScorer

WEB_SEARCH = [{"name": "web_search", "arguments": {"query": "weather in tokyo"}}]


def test_correct_call_should_score_one() -> None:
    sample = EvalSample(
        sample_id="p1",
        predicted_text=render_tool_calls([("web_search", {"query": "weather in tokyo"})]),
        expected_calls=WEB_SEARCH,
    )

    result = ToolCallScorer(arg_match="exact").score(sample)

    assert result.value == 1.0
    assert result.details["outcome"] == "correct_call"


def test_correct_abstention_should_score_one() -> None:
    result = ToolCallScorer().score(EvalSample(sample_id="n1", predicted_text="Hey, happy to help!", expected_calls=[]))

    assert result.value == 1.0


def test_diagnose_should_bind_the_lfm2_parser() -> None:
    diagnosis = diagnose("c", render_tool_calls([("db_query", {"q": "x"})]), WEB_SEARCH)

    assert diagnosis.outcome == "wrong_tool"
