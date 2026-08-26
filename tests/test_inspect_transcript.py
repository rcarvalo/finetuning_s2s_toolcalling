"""Tests for ``inspect_bridge.transcript.InspectTranscript``."""

from __future__ import annotations

import numpy as np
import pytest

from lfm2_audio.core.chat_format import render_tool_calls
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.scorer.sample import EvalSample

pytest.importorskip("inspect_ai")

from lfm2_audio.inspect_bridge.transcript import NO_ANSWER, InspectTranscript

CALL = render_tool_calls([("web_search", {"query": "weather Paris"})])


def _sample(**overrides: object) -> EvalSample:
    defaults: dict[str, object] = {
        "sample_id": "case_1",
        "prompt_text": "What is the weather in Paris?",
        "predicted_text": f"{CALL}It is sunny in Paris.",
        "tool_results": [
            {"name": "web_search", "arguments": {"query": "weather Paris"}, "ok": True, "result": "sunny"}
        ],
    }
    return EvalSample(**{**defaults, **overrides})  # type: ignore[arg-type]


def _roles(sample: EvalSample) -> list[str]:
    return [message.role for message in InspectTranscript(sample).messages()]


def test_should_render_the_whole_round_trip() -> None:
    """The reader must see the call and what the tool answered, not just the reply."""
    assert _roles(_sample()) == ["user", "assistant", "tool", "assistant"]


def test_should_carry_the_emitted_call_with_its_arguments() -> None:
    messages = InspectTranscript(_sample()).messages()

    call = messages[1].tool_calls[0]  # type: ignore[union-attr]
    assert call.function == "web_search"
    assert call.arguments == {"query": "weather Paris"}


def test_should_show_the_spoken_answer_without_the_call_span() -> None:
    """Leaving the span in makes `[web_search(...)]` look like the model's reply."""
    messages = InspectTranscript(_sample()).messages()

    text = messages[-1].content[0].text  # type: ignore[union-attr,index]
    assert text == "It is sunny in Paris."


def test_should_fall_back_to_a_marker_when_nothing_was_said() -> None:
    messages = InspectTranscript(_sample(predicted_text=CALL)).messages()

    assert messages[-1].content[0].text == NO_ANSWER  # type: ignore[union-attr,index]


def test_should_attach_the_audio_to_the_answer() -> None:
    audio = Waveform.of(np.zeros(2400, dtype=np.float32), 24_000)

    messages = InspectTranscript(_sample(predicted_audio=audio)).messages()

    parts = messages[-1].content
    assert [getattr(p, "type", "") for p in parts] == ["text", "audio"]  # type: ignore[union-attr]
    assert parts[1].audio.startswith("data:audio/wav;base64,")  # type: ignore[union-attr,index]


def test_should_omit_the_audio_when_there_is_none() -> None:
    parts = InspectTranscript(_sample()).messages()[-1].content

    assert [getattr(p, "type", "") for p in parts] == ["text"]  # type: ignore[union-attr]


def test_should_recover_calls_from_the_text_when_no_tool_ran() -> None:
    """A campaign that never executed tools still emitted the call."""
    messages = InspectTranscript(_sample(tool_results=[])).messages()

    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1].tool_calls[0].function == "web_search"  # type: ignore[union-attr,index]


def test_should_skip_the_tool_round_trip_when_no_call_was_made() -> None:
    sample = _sample(predicted_text="Hello there!", tool_results=[])

    assert _roles(sample) == ["user", "assistant"]


def test_should_mark_a_failed_tool_execution() -> None:
    sample = _sample(
        tool_results=[{"name": "web_search", "arguments": {"body": "x"}, "ok": False, "result": "missing query"}]
    )

    tool_message = InspectTranscript(sample).messages()[2]
    assert tool_message.error is not None
