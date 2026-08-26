"""``LiquidToolAgent`` — ReceptionAgent sous l'interface ``respond(wave)``.

Ce que ces tests protègent : la démo tool-calling doit pouvoir changer de
backend sans changer d'appelant. Le ChatState est porté par l'agent (et non
par l'UI), et ``reset`` doit vraiment ouvrir une conversation neuve — sinon le
contexte d'un tour fuit dans le suivant, ce qui est précisément le bug
multi-tours déjà payé une fois.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lfm2_audio.ds.audio import Waveform  # noqa: E402
from lfm2_audio.orchestrator.liquid_tool_agent import LiquidToolAgent  # noqa: E402


class _FakeReceptionAgent:
    def __init__(self) -> None:
        self.sessions = 0
        self.calls: list[tuple[Any, int]] = []
        self.registry = object()

    def new_session(self) -> str:
        self.sessions += 1
        return f"chat-{self.sessions}"

    def respond(self, chat: Any, samples: Any, rate: int):
        self.calls.append((chat, rate))
        yield f"event for {chat}"


def _wave(rate: int = 16_000) -> Waveform:
    return Waveform(np.zeros(rate // 2, dtype=np.float32), rate)


def test_should_open_a_session_on_construction() -> None:
    agent = _FakeReceptionAgent()

    LiquidToolAgent(agent)

    assert agent.sessions == 1


def test_should_reuse_the_same_session_across_turns() -> None:
    # Le multi-tours vit dans le ChatState : en ouvrir un par tour effacerait
    # le contexte et rendrait chaque relance incompréhensible.
    agent = _FakeReceptionAgent()
    tool_agent = LiquidToolAgent(agent)

    list(tool_agent.respond(_wave()))
    list(tool_agent.respond(_wave()))

    assert [chat for chat, _ in agent.calls] == ["chat-1", "chat-1"]


def test_should_open_a_new_session_on_reset() -> None:
    agent = _FakeReceptionAgent()
    tool_agent = LiquidToolAgent(agent)

    tool_agent.reset()
    list(tool_agent.respond(_wave()))

    assert agent.calls[0][0] == "chat-2"


def test_should_pass_the_waveform_sample_rate_through() -> None:
    # L'encodeur est calibré 16 kHz : perdre la fréquence dégrade sans erreur.
    agent = _FakeReceptionAgent()
    tool_agent = LiquidToolAgent(agent)

    list(tool_agent.respond(_wave(rate=24_000)))

    assert agent.calls[0][1] == 24_000


def test_should_forward_agent_events() -> None:
    agent = _FakeReceptionAgent()

    assert list(LiquidToolAgent(agent).respond(_wave())) == ["event for chat-1"]


def test_should_expose_the_tool_registry() -> None:
    agent = _FakeReceptionAgent()

    assert LiquidToolAgent(agent).registry is agent.registry
