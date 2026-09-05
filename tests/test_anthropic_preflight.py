"""Le pré-vol Anthropic passe par la MÊME forme de requête que le run (modèle, effort, streaming)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import anthropic
import pytest

PREFLIGHT = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "_anthropic_preflight.py"


@pytest.fixture
def preflight() -> Any:
    spec = importlib.util.spec_from_file_location("_anthropic_preflight", PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _Block:
    type: str = "text"
    text: str = "ok"


@dataclass
class _Usage:
    input_tokens: int = 5
    output_tokens: int = 1


@dataclass
class _Message:
    content: list[_Block] = field(default_factory=lambda: [_Block()])
    usage: _Usage = field(default_factory=_Usage)
    stop_reason: str = "end_turn"


class _Stream:
    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get_final_message(self) -> _Message:
        return _Message()


class _Messages:
    def __init__(self, owner: _FakeAnthropic) -> None:
        self._owner = owner

    def stream(self, **params: Any) -> _Stream:
        self._owner.calls.append(params)
        if self._owner.refuse:
            raise anthropic.AuthenticationError.__new__(anthropic.AuthenticationError)
        return _Stream()


class _FakeAnthropic:
    refuse = False
    instances: ClassVar[list[_FakeAnthropic]] = []

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.calls: list[dict[str, Any]] = []
        self.messages = _Messages(self)
        _FakeAnthropic.instances.append(self)


def test_should_make_one_tiny_call_with_the_run_shape(monkeypatch: pytest.MonkeyPatch, preflight: Any) -> None:
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc")  # pragma: allowlist secret

    preflight.preflight("claude-sonnet-5", "medium")

    sent = _FakeAnthropic.instances[-1].calls
    assert _FakeAnthropic.instances[-1].api_key == "sk-ant-abc"  # pragma: allowlist secret
    assert len(sent) == 1
    assert sent[0]["model"] == "claude-sonnet-5"
    assert sent[0]["output_config"] == {"effort": "medium"}
    assert sent[0]["max_tokens"] == 8


def test_should_exit_on_an_empty_key(monkeypatch: pytest.MonkeyPatch, preflight: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")

    with pytest.raises(SystemExit, match="manquant"):
        preflight.preflight("claude-opus-5")


def test_should_exit_with_model_and_key_shape_when_refused(monkeypatch: pytest.MonkeyPatch, preflight: Any) -> None:
    class _Refusing(_FakeAnthropic):
        refuse = True

    monkeypatch.setattr(anthropic, "Anthropic", _Refusing)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-wrong")  # pragma: allowlist secret

    with pytest.raises(SystemExit, match=r"claude-opus-5.*'sk-ant-', 12 caractères"):
        preflight.preflight("claude-opus-5")


def test_should_reject_an_unknown_effort_before_calling(monkeypatch: pytest.MonkeyPatch, preflight: Any) -> None:
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc")  # pragma: allowlist secret

    with pytest.raises(ValueError, match="effort inconnu"):
        preflight.preflight("claude-opus-5", "turbo")
