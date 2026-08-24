"""Contrat de fil client ↔ worker serverless (`lfm2_audio.remote.protocol`)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lfm2_audio.remote.protocol import (
    TURN_EVENT_ADAPTER,
    AudioChunkEvent,
    ErrorEvent,
    FinalEvent,
    JobEnvelope,
    StreamPage,
    TurnRequest,
)


class TestTurnRequest:
    def test_should_accept_text_only(self) -> None:
        request = TurnRequest(text="bonjour")

        assert request.text == "bonjour"
        assert request.audio_b64 is None

    def test_should_reject_request_without_text_nor_audio(self) -> None:
        with pytest.raises(ValidationError, match="text= ou audio="):
            TurnRequest()

    def test_should_reject_unknown_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            TurnRequest(text="hi", temperature=0.7)

    def test_should_reject_non_positive_max_tokens(self) -> None:
        with pytest.raises(ValidationError, match=r"greater[_ ]than"):
            TurnRequest(text="hi", max_tokens=0)

    def test_should_omit_none_fields_when_serialized(self) -> None:
        payload = TurnRequest(text="hi").model_dump(exclude_none=True, exclude_defaults=True)

        assert payload == {"text": "hi"}

    def test_should_accept_history_turns(self) -> None:
        request = TurnRequest.model_validate({"text": "hi", "history": [{"role": "assistant", "text": "earlier"}]})

        assert request.history[0].role == "assistant"
        assert request.history[0].text == "earlier"

    def test_should_reject_unknown_history_role(self) -> None:
        with pytest.raises(ValidationError):
            TurnRequest.model_validate({"text": "hi", "history": [{"role": "system", "text": "x"}]})


class TestTurnEvent:
    def test_should_discriminate_audio_event_on_kind(self) -> None:
        event = TURN_EVENT_ADAPTER.validate_python({"kind": "audio", "audio_b64": "AAA", "sample_rate": 24000})

        assert isinstance(event, AudioChunkEvent)
        assert event.sample_rate == 24_000

    def test_should_discriminate_final_event_and_default_its_metrics(self) -> None:
        event = TURN_EVENT_ADAPTER.validate_python({"kind": "final", "text": "salut"})

        assert isinstance(event, FinalEvent)
        assert event.metrics.audio_frames == 0
        assert event.metrics.ttfa_s is None

    def test_should_discriminate_error_event(self) -> None:
        event = TURN_EVENT_ADAPTER.validate_python({"kind": "error", "error": "OOM"})

        assert isinstance(event, ErrorEvent)
        assert event.error == "OOM"

    def test_should_reject_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            TURN_EVENT_ADAPTER.validate_python({"kind": "video", "url": "…"})

    def test_should_reject_audio_event_without_payload(self) -> None:
        with pytest.raises(ValidationError):
            TURN_EVENT_ADAPTER.validate_python({"kind": "audio", "sample_rate": 24000})


class TestJobEnvelope:
    def test_should_ignore_unknown_runpod_fields(self) -> None:
        envelope = JobEnvelope.model_validate({"id": "j1", "status": "COMPLETED", "delayTime": 42})

        assert envelope.id == "j1"

    def test_should_validate_each_output_event(self) -> None:
        envelope = JobEnvelope.model_validate(
            {"id": "j1", "status": "COMPLETED", "output": [{"kind": "final", "text": "ok"}]}
        )

        events = envelope.events

        assert len(events) == 1
        assert isinstance(events[0], FinalEvent)

    def test_should_return_no_event_when_output_is_absent(self) -> None:
        assert JobEnvelope(id="j1", status="IN_QUEUE").events == []

    def test_should_raise_when_output_is_not_a_list(self) -> None:
        envelope = JobEnvelope.model_validate({"id": "j1", "status": "COMPLETED", "output": {"oops": True}})

        with pytest.raises(ValueError, match="sortie inattendue"):
            _ = envelope.events


class TestStreamPage:
    def test_should_parse_stream_items(self) -> None:
        page = StreamPage.model_validate(
            {
                "status": "IN_PROGRESS",
                "stream": [{"output": {"kind": "audio", "audio_b64": "AA", "sample_rate": 24000}}],
            }
        )

        assert isinstance(page.stream[0].output, AudioChunkEvent)

    def test_should_default_to_empty_stream(self) -> None:
        assert StreamPage().stream == []
