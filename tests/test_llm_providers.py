"""Le fournisseur d'un générateur se choisit par un drapeau, jamais par un import."""

from __future__ import annotations

import pytest

from lfm2_audio.cli.data.llm_providers import judge_stream, make_judge, spend_line
from lfm2_audio.scorer.text.anthropic_batch_judge import AnthropicBatchJudge
from lfm2_audio.scorer.text.anthropic_judge import AnthropicJudge
from lfm2_audio.scorer.text.gemini_judge import GeminiJudge


class TestMakeJudge:
    def test_should_build_a_gemini_judge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "AQ.x")  # pragma: allowlist secret

        assert isinstance(make_judge("gemini"), GeminiJudge)

    def test_should_build_a_streaming_anthropic_judge_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")  # pragma: allowlist secret

        judge = make_judge("anthropic", max_usd=2.0)

        assert isinstance(judge, AnthropicJudge)
        assert judge.model_id == "claude-opus-5"
        assert judge.meter.max_usd == 2.0

    def test_should_build_a_batch_judge_on_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")  # pragma: allowlist secret

        judge = make_judge("anthropic", "claude-sonnet-5", batch=True, max_usd=3.0)

        assert isinstance(judge, AnthropicBatchJudge)
        assert judge.model_id == "claude-sonnet-5"
        assert judge.meter.max_usd == 3.0

    def test_should_name_the_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY absent"):
            make_judge("anthropic")

    def test_should_reject_an_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="fournisseur inconnu"):
            make_judge("openai")


class _Sequential:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def judge(self, prompt: str) -> str:
        self.seen.append(prompt)
        return prompt.upper()


class _Batched(_Sequential):
    def judge_many(self, prompts: list[str]) -> list[str]:
        self.seen.append("lot")
        return [p.upper() for p in prompts]


class TestJudgeStream:
    def test_should_call_a_plain_judge_once_per_prompt(self) -> None:
        judge = _Sequential()

        assert list(judge_stream(judge, ["a", "b"])) == ["A", "B"]
        assert judge.seen == ["a", "b"]

    def test_should_use_the_batch_path_when_the_judge_offers_it(self) -> None:
        judge = _Batched()

        assert list(judge_stream(judge, ["a", "b"])) == ["A", "B"]
        assert judge.seen == ["lot"]


class TestSpendLine:
    def test_should_be_absent_for_a_judge_that_does_not_count(self) -> None:
        assert spend_line(_Sequential()) is None

    def test_should_report_the_meter_of_a_counting_judge(self) -> None:
        judge = AnthropicJudge("claude-sonnet-5", api_key="sk-ant-x")  # pragma: allowlist secret

        assert str(spend_line(judge)).startswith("===SPEND=== model=claude-sonnet-5")
