"""``HfJudge`` — juge LLM adossé aux Inference Providers Hugging Face.

Exists because the project has an HF token but no Gemini key. Without a
fallback judge, the `reasoning` scorer stays UNAVAILABLE and a campaign
measures grounding while never measuring relevance — the blind spot that let
v3 ship an answer that quoted the payload without answering the question.
"""

from __future__ import annotations

from typing import Any

import pytest

from lfm2_audio.scorer.text.hf_judge import DEFAULT_MODEL_ID, HfJudge
from lfm2_audio.scorer.text.judge import Judge


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("Msg", (), {"content": content})()


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return type("Resp", (), {"choices": [_FakeChoice('{"scores": {"relevance": 4}}')]})()


def test_should_satisfy_the_judge_protocol() -> None:
    assert isinstance(HfJudge(api_key="k"), Judge)


def test_should_report_missing_credentials() -> None:
    assert HfJudge(api_key="").has_credentials is False
    assert HfJudge(api_key="hf_x").has_credentials is True  # pragma: allowlist secret


def test_should_read_the_token_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")  # pragma: allowlist secret

    assert HfJudge().has_credentials is True


def test_should_return_the_model_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = HfJudge(api_key="k")
    client = _FakeClient()
    monkeypatch.setattr(judge, "_inference", lambda: client)

    assert judge.judge("grade this") == '{"scores": {"relevance": 4}}'


def test_should_judge_deterministically_on_the_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Temperature 0: a rubric score that moves between runs is not a metric.
    judge = HfJudge(api_key="k")
    client = _FakeClient()
    monkeypatch.setattr(judge, "_inference", lambda: client)

    judge.judge("grade this")

    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["model"] == DEFAULT_MODEL_ID
