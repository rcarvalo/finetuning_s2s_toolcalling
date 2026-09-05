"""``AnthropicJudge`` — un prompt, une réponse, l'usage compté, sans réseau."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from lfm2_audio.scorer.text.anthropic_judge import AnthropicJudge, parse_effort, text_of
from lfm2_audio.scorer.text.judge import Judge
from lfm2_audio.scorer.text.llm_spend import SpendCapReachedError, SpendMeter


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
class _Stream:
    message: _Message

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get_final_message(self) -> _Message:
        return self.message


@dataclass
class _FakeMessages:
    reply: _Message
    calls: list[dict[str, Any]] = field(default_factory=list)

    def stream(self, **params: Any) -> _Stream:
        self.calls.append(params)
        return _Stream(self.reply)


@dataclass
class _FakeClient:
    messages: _FakeMessages


def _judge_with(reply: _Message, meter: SpendMeter | None = None) -> tuple[AnthropicJudge, _FakeMessages]:
    judge = AnthropicJudge("claude-sonnet-5", api_key="sk-ant-test", meter=meter)  # pragma: allowlist secret
    messages = _FakeMessages(reply)
    judge._client = _FakeClient(messages)  # type: ignore[assignment]
    return judge, messages


class TestAnthropicJudge:
    def test_should_satisfy_the_judge_protocol(self) -> None:
        assert isinstance(AnthropicJudge("claude-opus-5", api_key="sk-ant-x"), Judge)  # pragma: allowlist secret

    def test_should_report_credentials_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")  # pragma: allowlist secret

        assert AnthropicJudge("claude-opus-5").has_credentials

    def test_should_report_missing_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        assert not AnthropicJudge("claude-opus-5").has_credentials

    def test_should_return_the_text_and_count_the_usage(self) -> None:
        reply = _Message([_Block("text", "[{}]")], _Usage(500, 5_000))
        judge, _ = _judge_with(reply)

        text = judge.judge("génère")

        assert text == "[{}]"
        assert (judge.meter.input_tokens, judge.meter.output_tokens, judge.meter.calls) == (500, 5_000, 1)

    def test_should_send_the_model_effort_and_prompt(self) -> None:
        judge, messages = _judge_with(_Message([_Block("text", "ok")], _Usage(1, 1)))

        judge.judge("génère")

        sent = messages.calls[0]
        assert sent["model"] == "claude-sonnet-5"
        assert sent["output_config"] == {"effort": "low"}
        assert sent["messages"] == [{"role": "user", "content": "génère"}]
        assert "stream" not in sent

    def test_should_refuse_to_call_once_the_cap_is_reached(self) -> None:
        meter = SpendMeter("claude-sonnet-5", max_usd=0.001)
        meter.add(0, 1_000)  # 0,01 $ > plafond
        judge, messages = _judge_with(_Message([_Block("text", "ok")], _Usage(1, 1)), meter)

        with pytest.raises(SpendCapReachedError):
            judge.judge("génère")
        assert messages.calls == []


class TestTextOf:
    def test_should_join_only_text_blocks(self) -> None:
        message = _Message([_Block("thinking"), _Block("text", "a"), _Block("text", "b")], _Usage(1, 1))

        assert text_of(message) == "ab"  # type: ignore[arg-type]

    def test_should_drop_a_refused_answer(self) -> None:
        message = _Message([_Block("text", "partiel")], _Usage(1, 1), stop_reason="refusal")

        assert text_of(message) == ""  # type: ignore[arg-type]

    def test_should_drop_a_truncated_answer(self) -> None:
        # Un JSON coupé ne se répare pas ; mieux vaut un lot vide et un avertissement.
        message = _Message([_Block("text", '[{"turns": [')], _Usage(1, 1), stop_reason="max_tokens")

        assert text_of(message) == ""  # type: ignore[arg-type]


class TestParseEffort:
    def test_should_accept_a_known_level(self) -> None:
        assert parse_effort("medium") == "medium"

    def test_should_reject_an_unknown_level(self) -> None:
        with pytest.raises(ValueError, match="effort inconnu"):
            parse_effort("turbo")
