"""Tests du scorer de tool calling — en particulier le traitement des négatifs."""

from __future__ import annotations

import pytest

from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.status import ScoreStatus
from lfm2_audio.scorer.text.tool_call import ToolCallScorer

WEB_SEARCH = [{"name": "web_search", "arguments": {"query": "weather in tokyo"}}]


def _span(name: str, **arguments: str) -> str:
    rendered = ", ".join(f'{k}="{v}"' for k, v in arguments.items())
    return f"<|tool_call_start|>[{name}({rendered})]<|tool_call_end|>"


@pytest.fixture
def scorer() -> ToolCallScorer:
    return ToolCallScorer()


def test_correct_call_should_score_one(scorer):
    sample = EvalSample(
        sample_id="p1",
        predicted_text=_span("web_search", query="weather in tokyo"),
        expected_calls=WEB_SEARCH,
    )

    result = scorer.score(sample)

    assert result.value == 1.0
    assert result.details["name"] is True
    assert result.details["call"] is True


def test_wrong_tool_should_score_zero(scorer):
    sample = EvalSample(
        sample_id="p2",
        predicted_text=_span("db_query", question="weather in tokyo"),
        expected_calls=WEB_SEARCH,
    )

    result = scorer.score(sample)

    assert result.value == 0.0
    assert result.details["relevance"] is True  # appeler était la bonne décision
    assert result.details["name"] is False


def test_correct_abstention_should_score_one(scorer):
    """`call_correct` est faux par construction sur un négatif : agréger cette
    facette seule punirait chaque abstention correcte."""
    sample = EvalSample(sample_id="n1", predicted_text="Hey, happy to help!", expected_calls=[])

    result = scorer.score(sample)

    assert result.value == 1.0
    assert result.details["call"] is False
    assert result.details["relevance"] is True


def test_unwarranted_call_should_score_zero(scorer):
    sample = EvalSample(sample_id="n2", predicted_text=_span("web_search", query="hi"), expected_calls=[])

    result = scorer.score(sample)

    assert result.value == 0.0
    assert result.details["relevance"] is False


def test_missing_call_should_score_zero(scorer):
    sample = EvalSample(sample_id="p3", predicted_text="I don't know.", expected_calls=WEB_SEARCH)

    result = scorer.score(sample)

    assert result.value == 0.0
    assert result.details["relevance"] is False


def test_should_report_a_parse_failure(scorer):
    sample = EvalSample(
        sample_id="p4",
        predicted_text="<|tool_call_start|>[web_search(query=<|tool_call_end|>",
        expected_calls=WEB_SEARCH,
    )

    result = scorer.score(sample)

    assert result.details["parse"] is False
    # Une tentative malformée reste une tentative : la pertinence est sauve.
    assert result.details["predicted_call"] is True


def test_should_skip_an_empty_generation(scorer):
    result = scorer.score(EvalSample(sample_id="e1", expected_calls=WEB_SEARCH))

    assert result.status is ScoreStatus.SKIPPED


def test_tolerant_arg_matching_should_accept_a_paraphrase():
    tolerant = ToolCallScorer(arg_match="token_f1", threshold=0.5)
    strict = ToolCallScorer(arg_match="exact")
    sample = EvalSample(
        sample_id="p5",
        predicted_text=_span("web_search", query="weather tokyo"),
        expected_calls=WEB_SEARCH,
    )

    assert tolerant.score(sample).value == 1.0
    assert strict.score(sample).value == 0.0
