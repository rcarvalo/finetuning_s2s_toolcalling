"""Tests de l'orchestration 2-passes ``VllmToolAgent`` (backend mocké, sans GPU)."""

from __future__ import annotations

import numpy as np

from s2s_toolcalling.data import chat_format
from s2s_toolcalling.orchestrator.events import (
    AudioChunk,
    FillerSpeech,
    ToolCallBegin,
    ToolCallResult,
    TurnComplete,
)
from s2s_toolcalling.orchestrator.vllm_tool_agent import Turn, VllmToolAgent
from s2s_toolcalling.tools.toolcalling_en import StubDbQueryBackend, build_toolcalling_en_registry
from s2s_toolcalling.tools.web_search import StubWebSearchBackend


class FakeBackend:
    """Rejoue une file de réponses ``(texte, n_chunks)`` à chaque ``stream``."""

    def __init__(self, scripted: list[tuple[str, int]]) -> None:
        self._scripted = list(scripted)
        self.system = ""
        self.last_text = ""
        self.tool_call_end_id = 42
        self.im_end_id = 7
        self.calls: list[list[Turn]] = []

    def stream(self, turns, *, stop_token_ids):
        # snapshot du contexte vu à cette passe (copie pour figer le texte user)
        self.calls.append([Turn(t.role, t.text, t.audio) for t in turns])
        text, n_chunks = self._scripted.pop(0)
        self.last_text = text
        for _ in range(n_chunks):
            yield np.zeros(1920, dtype=np.float32)


def _registry():
    return build_toolcalling_en_registry(
        web_backend=StubWebSearchBackend(), db_backend=StubDbQueryBackend()
    )


def _agent(scripted):
    return VllmToolAgent(FakeBackend(scripted), _registry())


def test_system_prompt_includes_instructions_and_tools():
    agent = _agent([])
    assert chat_format.TOOLCALLING_EN_SYSTEM_INSTRUCTIONS in agent.backend.system
    assert chat_format.TOOL_LIST_START in agent.backend.system


def test_negative_single_pass_streams_answer():
    agent = _agent([("Hi there, happy to help!", 3)])

    events = list(agent.respond(np.zeros(16_000, dtype=np.float32), 16_000))

    assert sum(isinstance(e, AudioChunk) for e in events) == 3
    assert not any(isinstance(e, ToolCallBegin) for e in events)
    done = events[-1]
    assert isinstance(done, TurnComplete) and done.tool_rounds == 0
    assert done.text == "Hi there, happy to help!"


def test_positive_runs_tool_then_speaks_answer():
    call_span = '<|tool_call_start|>[web_search(query="weather in Paris")]<|tool_call_end|>'
    agent = _agent([(call_span, 0), ("It's sunny in Paris.", 4)])

    events = list(agent.respond(np.zeros(16_000, dtype=np.float32), 16_000))

    begin = next(e for e in events if isinstance(e, ToolCallBegin))
    assert begin.name == "web_search" and begin.arguments == {"query": "weather in Paris"}
    assert any(isinstance(e, FillerSpeech) for e in events)
    res = next(e for e in events if isinstance(e, ToolCallResult))
    assert res.ok
    assert sum(isinstance(e, AudioChunk) for e in events) == 4  # Pass B uniquement
    done = events[-1]
    assert isinstance(done, TurnComplete) and done.tool_rounds == 1
    assert done.text == "It's sunny in Paris."


def test_positive_pass_b_drops_user_audio_and_has_tool_turn():
    call_span = '<|tool_call_start|>[web_search(query="weather")]<|tool_call_end|>'
    agent = _agent([(call_span, 0), ("Sunny.", 2)])

    list(agent.respond(np.zeros(16_000, dtype=np.float32), 16_000))

    pass_a, pass_b = agent.backend.calls
    # Pass A : tour user porte l'audio (dernière position).
    assert pass_a[-1].role == "user" and pass_a[-1].audio is not None
    # Pass B : audio consommé (texte « (voice message) ») + tour tool injecté.
    user_b = next(t for t in pass_b if t.role == "user")
    assert user_b.audio is None and user_b.text == "(voice message)"
    roles_b = [t.role for t in pass_b]
    assert roles_b == ["user", "assistant", "tool"]
    assert chat_format.TOOL_RESPONSE_START in pass_b[-1].text


def test_stop_token_stripped_still_parses_call():
    # vLLM strippe le stop token <|tool_call_end|> : le span est ouvert sans fermeture.
    open_span = '<|tool_call_start|>[db_query(question="how many orders")]'
    agent = _agent([(open_span, 0), ("Five orders.", 1)])

    events = list(agent.respond(np.zeros(16_000, dtype=np.float32), 16_000))

    begin = next(e for e in events if isinstance(e, ToolCallBegin))
    assert begin.name == "db_query"
