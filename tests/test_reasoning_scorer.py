"""Tests du scorer de raisonnement (juge factice, sans appel réseau)."""

from __future__ import annotations

import json

from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.status import ScoreStatus
from lfm2_audio.scorer.text.reasoning import ReasoningScorer
from lfm2_audio.scorer.text.rubric import REASONING_RUBRIC, JudgeCriterion, JudgeRubric


class FakeJudge:
    """Rend un verdict fixé. Satisfait le protocole ``Judge``."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def judge(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


def _verdict(**scores: float) -> str:
    return json.dumps({"scores": scores, "rationale": "fine"})


def _sample(**kwargs) -> EvalSample:
    return EvalSample(
        sample_id="s1",
        prompt_text="What's the weather in Tokyo?",
        predicted_text="It's 18 degrees and sunny in Tokyo.",
        **kwargs,
    )


def test_perfect_scores_should_reach_one():
    judge = FakeJudge(_verdict(relevance=5, grounding=5, coherence=5, conciseness=5))

    result = ReasoningScorer(judge).score(_sample())

    assert result.value == 1.0
    assert result.details["rubric_version"] == REASONING_RUBRIC.version


def test_lowest_scores_should_reach_a_fifth():
    judge = FakeJudge(_verdict(relevance=1, grounding=1, coherence=1, conciseness=1))

    assert ReasoningScorer(judge).score(_sample()).value == 0.2


def test_grounding_should_weigh_more_than_conciseness():
    """L'ancrage est le mode d'échec dominant : la rubrique le pondère davantage."""
    weak_grounding = FakeJudge(_verdict(relevance=5, grounding=1, coherence=5, conciseness=5))
    weak_conciseness = FakeJudge(_verdict(relevance=5, grounding=5, coherence=5, conciseness=1))

    assert (
        ReasoningScorer(weak_grounding).score(_sample()).value
        < ReasoningScorer(weak_conciseness).score(_sample()).value
    )


def test_missing_criteria_should_be_ignored_not_counted_zero():
    partial = FakeJudge(_verdict(relevance=5))

    assert ReasoningScorer(partial).score(_sample()).value == 1.0


def test_should_tolerate_markdown_fences_around_the_json():
    judge = FakeJudge("```json\n" + _verdict(relevance=4, grounding=4) + "\n```")

    assert ReasoningScorer(judge).score(_sample()).value == 0.8


def test_unparsable_verdict_should_fail_not_crash():
    result = ReasoningScorer(FakeJudge("I cannot comply.")).score(_sample())

    assert result.status is ScoreStatus.FAILED
    assert "inexploitable" in result.reason


def test_prompt_should_carry_question_answer_and_tool_result():
    judge = FakeJudge(_verdict(relevance=3))
    sample = _sample(tool_results=[{"temperature": "18C"}])

    ReasoningScorer(judge).score(sample)

    prompt = judge.prompts[0]
    assert "What's the weather in Tokyo?" in prompt
    assert "18C" in prompt
    assert "It's 18 degrees and sunny in Tokyo." in prompt


def test_prompt_should_say_when_no_tool_was_called():
    judge = FakeJudge(_verdict(relevance=3))

    ReasoningScorer(judge).score(_sample())

    assert "no tool was called" in judge.prompts[0]


def test_should_skip_a_pure_tool_call_turn():
    """Un tour qui n'émet qu'un appel n'a pas de réponse parlée à juger."""
    judge = FakeJudge(_verdict(relevance=5))
    sample = EvalSample(sample_id="s2", predicted_text="<|tool_call_start|>[web_search(query='x')]<|tool_call_end|>")

    result = ReasoningScorer(judge).score(sample)

    assert result.status is ScoreStatus.SKIPPED
    assert judge.prompts == []


def test_custom_rubric_should_drive_the_scoring():
    rubric = JudgeRubric(version="single-v1", criteria=(JudgeCriterion(key="only", question="Good?"),), scale=10)
    judge = FakeJudge(_verdict(only=7))

    result = ReasoningScorer(judge, rubric=rubric).score(_sample())

    assert result.value == 0.7
    assert result.details["rubric_version"] == "single-v1"
