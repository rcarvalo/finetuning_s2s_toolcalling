"""``AnthropicBatchJudge`` — N prompts, un lot, les réponses remises dans l'ordre."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from lfm2_audio.scorer.text.anthropic_batch_judge import AnthropicBatchJudge
from lfm2_audio.scorer.text.judge import Judge
from lfm2_audio.scorer.text.llm_spend import SpendCapReachedError


@dataclass
class _Block:
    type: str
    text: str = ""


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Message:
    content: list[_Block]
    usage: _Usage
    stop_reason: str = "end_turn"


@dataclass
class _Outcome:
    type: str
    message: _Message | None = None


@dataclass
class _Result:
    custom_id: str
    result: _Outcome


@dataclass
class _Counts:
    processing: int


@dataclass
class _Batch:
    id: str
    processing_status: str
    request_counts: _Counts = field(default_factory=lambda: _Counts(0))


class _FakeBatches:
    """Un lot qui passe par `in_progress` une fois avant `ended`, résultats en désordre."""

    def __init__(self, results: list[_Result]) -> None:
        self._results = results
        self.created: list[list[Any]] = []
        self.polls = 0

    def create(self, requests: list[Any]) -> _Batch:
        self.created.append(requests)
        return _Batch("msgbatch_1", "in_progress", _Counts(len(requests)))

    def retrieve(self, batch_id: str) -> _Batch:
        self.polls += 1
        return _Batch(batch_id, "ended" if self.polls > 1 else "in_progress")

    def results(self, batch_id: str) -> list[_Result]:
        return list(reversed(self._results))


@dataclass
class _FakeMessages:
    batches: _FakeBatches


@dataclass
class _FakeClient:
    messages: _FakeMessages


def _judge_with(
    results: list[_Result], max_usd: float | None = None
) -> tuple[AnthropicBatchJudge, _FakeBatches, list[float]]:
    naps: list[float] = []
    judge = AnthropicBatchJudge(
        "claude-sonnet-5",
        api_key="sk-ant-test",  # pragma: allowlist secret
        max_usd=max_usd,
        poll_seconds=7.0,
        sleep=naps.append,
    )
    batches = _FakeBatches(results)
    judge._single._client = _FakeClient(_FakeMessages(batches))  # type: ignore[assignment]
    return judge, batches, naps


def _ok(index: int, text: str, out_tokens: int = 1_000) -> _Result:
    return _Result(f"p{index}", _Outcome("succeeded", _Message([_Block("text", text)], _Usage(100, out_tokens))))


class TestAnthropicBatchJudge:
    def test_should_satisfy_the_judge_protocol(self) -> None:
        assert isinstance(AnthropicBatchJudge("claude-opus-5", api_key="sk-ant-x"), Judge)  # pragma: allowlist secret

    def test_should_submit_one_request_per_prompt_with_the_shared_params(self) -> None:
        judge, batches, _ = _judge_with([_ok(0, "a"), _ok(1, "b")])

        judge.judge_many(["un", "deux"])

        requests = batches.created[0]
        assert [r["custom_id"] for r in requests] == ["p0", "p1"]
        assert requests[1]["params"]["messages"] == [{"role": "user", "content": "deux"}]
        assert requests[1]["params"]["output_config"] == {"effort": "low"}

    def test_should_return_the_answers_in_prompt_order_whatever_the_result_order(self) -> None:
        judge, _, _ = _judge_with([_ok(0, "a"), _ok(1, "b"), _ok(2, "c")])

        assert judge.judge_many(["1", "2", "3"]) == ["a", "b", "c"]

    def test_should_wait_between_polls_until_the_batch_ends(self) -> None:
        judge, batches, naps = _judge_with([_ok(0, "a")])

        judge.judge_many(["1"])

        assert batches.polls == 2
        assert naps == [7.0]

    def test_should_charge_at_half_price(self) -> None:
        judge, _, _ = _judge_with([_ok(0, "a", out_tokens=1_000_000)])  # 10 $ plein tarif Sonnet 5

        judge.judge_many(["1"])

        assert judge.meter.usd == pytest.approx(5.0, abs=0.001)

    def test_should_leave_a_failed_request_empty_without_losing_the_others(self) -> None:
        judge, _, _ = _judge_with([_ok(0, "a"), _Result("p1", _Outcome("errored")), _ok(2, "c")])

        assert judge.judge_many(["1", "2", "3"]) == ["a", "", "c"]

    def test_should_answer_a_single_prompt_through_the_same_path(self) -> None:
        judge, batches, _ = _judge_with([_ok(0, "seul")])

        assert judge.judge("1") == "seul"
        assert len(batches.created) == 1

    def test_should_not_submit_anything_once_the_cap_is_reached(self) -> None:
        judge, batches, _ = _judge_with([_ok(0, "a")], max_usd=0.0001)
        judge.meter.add(0, 1_000)

        with pytest.raises(SpendCapReachedError):
            judge.judge_many(["1"])
        assert batches.created == []

    def test_should_return_nothing_for_no_prompts_without_a_call(self) -> None:
        judge, batches, _ = _judge_with([])

        assert judge.judge_many([]) == []
        assert batches.created == []
