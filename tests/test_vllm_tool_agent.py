"""Tests de l'orchestration 2-passes ``VllmToolAgent`` (backend mocké, sans GPU)."""

from __future__ import annotations

import numpy as np

from lfm2_audio.core import chat_format
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.orchestrator.events import (
    AudioChunk,
    FillerSpeech,
    ToolCallBegin,
    ToolCallResult,
    TurnComplete,
)
from lfm2_audio.orchestrator.vllm_tool_agent import Turn, VllmToolAgent
from lfm2_audio.tools.toolcalling_en import StubDbQueryBackend, build_toolcalling_en_registry
from lfm2_audio.tools.web_search import StubWebSearchBackend


class FakeBackend:
    """Rejoue une file de réponses ``(texte, n_chunks)`` à chaque ``stream_turns``."""

    def __init__(self, scripted: list[tuple[str, int]]) -> None:
        self._scripted = list(scripted)
        self.system = ""
        self.last_text = ""
        self.tool_call_end_id = 42
        self.im_end_id = 7
        self.calls: list[list[Turn]] = []

    def stream_turns(self, turns, *, stop_token_ids):
        # snapshot du contexte vu à cette passe (copie pour figer le texte user)
        self.calls.append([Turn(t.role, t.text, t.audio) for t in turns])
        text, n_chunks = self._scripted.pop(0)
        self.last_text = text
        for _ in range(n_chunks):
            yield Waveform.of(np.zeros(1920, dtype=np.float32), 24_000)


def _silence(seconds: float = 1.0) -> Waveform:
    return Waveform.of(np.zeros(int(16_000 * seconds), dtype=np.float32), 16_000)


def _registry():
    return build_toolcalling_en_registry(web_backend=StubWebSearchBackend(), db_backend=StubDbQueryBackend())


def _agent(scripted):
    return VllmToolAgent(FakeBackend(scripted), _registry())


def test_system_prompt_includes_instructions_and_tools():
    agent = _agent([])
    assert chat_format.TOOLCALLING_EN_SYSTEM_INSTRUCTIONS in agent.backend.system
    assert chat_format.TOOL_LIST_START in agent.backend.system


def test_negative_single_pass_streams_answer():
    agent = _agent([("Hi there, happy to help!", 3)])

    events = list(agent.respond(_silence()))

    assert sum(isinstance(e, AudioChunk) for e in events) == 3
    assert not any(isinstance(e, ToolCallBegin) for e in events)
    done = events[-1]
    assert isinstance(done, TurnComplete) and done.tool_rounds == 0
    assert done.text == "Hi there, happy to help!"


def test_positive_runs_tool_then_speaks_answer():
    call_span = '<|tool_call_start|>[web_search(query="weather in Paris")]<|tool_call_end|>'
    agent = _agent([(call_span, 0), ("It's sunny in Paris.", 4)])

    events = list(agent.respond(_silence()))

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

    list(agent.respond(_silence()))

    pass_a, pass_b = agent.backend.calls
    # Pass A : tour user porte l'audio (dernière position).
    assert pass_a[-1].role == "user" and pass_a[-1].audio is not None
    # Pass B : audio consommé (texte « (voice message) ») + tour tool injecté.
    user_b = next(t for t in pass_b if t.role == "user")
    assert user_b.audio is None and user_b.text == "(voice message)"
    roles_b = [t.role for t in pass_b]
    assert roles_b == ["user", "assistant", "tool"]
    assert chat_format.TOOL_RESPONSE_START in pass_b[-1].text


def test_visible_strips_audio_placeholders_from_text_and_history():
    # Le flux brut interleavé contient des <|audio_start|> : ils DOIVENT être retirés
    # du texte parlable ET de l'historique (sinon re-tokenisés en placeholder audio
    # sans frame → contexte corrompu au tour suivant, « I'm not able to help »).
    raw = "On June 23, <|audio_start|><|audio_start|>2026, sunny.<|text_end|><|audio_start|><|text_start|>"
    agent = _agent([(raw, 5)])

    events = list(agent.respond(_silence()))

    done = events[-1]
    assert done.text == "On June 23, 2026, sunny."
    assert "<|" not in agent.turns[-1].text  # historique propre


def test_stop_token_stripped_still_parses_call():
    # vLLM strippe le stop token <|tool_call_end|> : le span est ouvert sans fermeture.
    open_span = '<|tool_call_start|>[db_query(question="how many orders")]'
    agent = _agent([(open_span, 0), ("Five orders.", 1)])

    events = list(agent.respond(_silence()))

    begin = next(e for e in events if isinstance(e, ToolCallBegin))
    assert begin.name == "db_query"
