"""``LiquidAudioClient`` — transport mocké (httpx.MockTransport), zéro réseau."""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest

from lfm2_audio.core.errors import RemoteInferenceError
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.remote.client import LiquidAudioClient
from lfm2_audio.remote.wav_base64 import waveform_to_wav_b64

ENDPOINT = "ep123"


def _audio_event(value: float = 0.1, n: int = 160) -> dict[str, Any]:
    wave = Waveform.of(np.full(n, value, dtype=np.float32), 24_000)
    return {"kind": "audio", "audio_b64": waveform_to_wav_b64(wave), "sample_rate": 24_000}


def _final_event(text: str = "bonjour") -> dict[str, Any]:
    return {
        "kind": "final",
        "text": text,
        "raw_text": f"{text}<|eot|>",
        "metrics": {"ttfa_s": 0.4, "total_s": 1.2, "audio_frames": 2},
    }


def _client(handler: Any) -> LiquidAudioClient:
    return LiquidAudioClient(
        ENDPOINT,
        api_key="rp_test",  # pragma: allowlist secret — test fixture
        poll_interval_s=0.0,
        transport=httpx.MockTransport(handler),
    )


def _routes(responses: dict[str, list[dict[str, Any]] | dict[str, Any]]) -> Any:
    """Handler MockTransport : chemin (relatif à l'endpoint) → réponse(s) successives."""

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path.split(f"/{ENDPOINT}", 1)[1]
        planned = responses[path]
        body = planned.pop(0) if isinstance(planned, list) else planned
        return httpx.Response(200, json=body)

    return handle


def test_should_require_api_key_when_env_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    with pytest.raises(RemoteInferenceError, match="RUNPOD_API_KEY"):
        LiquidAudioClient(ENDPOINT)


def test_should_reject_invoke_without_text_nor_audio() -> None:
    client = _client(_routes({}))

    with pytest.raises(RemoteInferenceError, match="text= ou audio="):
        client.invoke()


def test_should_return_reply_with_concatenated_audio_on_runsync() -> None:
    completed = {
        "id": "job1",
        "status": "COMPLETED",
        "output": [_audio_event(0.1), _audio_event(0.2), _final_event()],
    }
    client = _client(_routes({"/runsync": completed}))

    text, audio = client.invoke(text="salut")

    assert text == "bonjour"
    assert audio is not None
    assert audio.samples.shape == (320,)
    assert client.last_reply is not None
    assert client.last_reply.metrics.audio_frames == 2
    assert client.last_reply.raw_text == "bonjour<|eot|>"


def test_should_send_audio_as_b64_in_input_payload() -> None:
    captured: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"id": "j", "status": "COMPLETED", "output": [_final_event()]})

    client = _client(handle)
    client.invoke(audio=Waveform.of(np.zeros(160, dtype=np.float32), 16_000), max_tokens=64)

    assert set(captured["input"]) == {"audio_b64", "max_tokens"}


def test_should_poll_status_when_runsync_times_out_in_progress() -> None:
    client = _client(
        _routes(
            {
                "/runsync": {"id": "job2", "status": "IN_PROGRESS"},
                "/status/job2": [
                    {"id": "job2", "status": "IN_PROGRESS"},
                    {"id": "job2", "status": "COMPLETED", "output": [_final_event("fini")]},
                ],
            }
        )
    )

    reply = client.invoke(text="salut")

    assert reply.text == "fini"


def test_should_raise_typed_error_when_job_fails() -> None:
    failed = {"id": "job3", "status": "FAILED", "error": "CUDA OOM"}
    client = _client(_routes({"/runsync": failed}))

    with pytest.raises(RemoteInferenceError, match="CUDA OOM"):
        client.invoke(text="salut")


def test_should_raise_typed_error_on_http_failure() -> None:
    client = _client(lambda request: httpx.Response(401, json={"error": "unauthorized"}))

    with pytest.raises(RemoteInferenceError, match="/runsync"):
        client.invoke(text="salut")


def test_should_yield_chunks_then_expose_last_reply_on_stream() -> None:
    client = _client(
        _routes(
            {
                "/run": {"id": "job4", "status": "IN_QUEUE"},
                "/stream/job4": [
                    {"status": "IN_PROGRESS", "stream": [{"output": _audio_event(0.1)}]},
                    {
                        "status": "COMPLETED",
                        "stream": [{"output": _audio_event(0.2)}, {"output": _final_event("streamé")}],
                    },
                ],
            }
        )
    )

    chunks = list(client.invoke_stream(text="salut"))

    assert len(chunks) == 2
    assert all(chunk.sample_rate == 24_000 for chunk in chunks)
    assert client.last_reply is not None
    assert client.last_reply.text == "streamé"
    assert client.last_reply.audio is not None
    assert client.last_reply.audio.samples.shape == (320,)


def test_should_raise_on_worker_error_event_in_stream() -> None:
    client = _client(
        _routes(
            {
                "/run": {"id": "job5", "status": "IN_QUEUE"},
                "/stream/job5": {
                    "status": "COMPLETED",
                    "stream": [{"output": {"kind": "error", "error": "champ inconnu : history"}}],
                },
            }
        )
    )

    with pytest.raises(RemoteInferenceError, match="champ inconnu"):
        list(client.invoke_stream(text="salut"))


def test_should_raise_on_worker_error_event_in_runsync() -> None:
    client = _client(
        _routes(
            {
                "/runsync": {
                    "id": "job6",
                    "status": "COMPLETED",
                    "output": [{"kind": "error", "error": "boom"}],
                }
            }
        )
    )

    with pytest.raises(RemoteInferenceError, match="boom"):
        client.invoke(text="salut")


def test_should_send_history_in_input_payload() -> None:
    seen: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"id": "job7", "status": "COMPLETED", "output": [_final_event("ok")]})

    client = _client(handle)

    client.invoke(text="hello", history=[("user", ""), ("assistant", "earlier reply")])

    assert seen["input"]["history"] == [
        {"role": "user", "text": ""},
        {"role": "assistant", "text": "earlier reply"},
    ]
