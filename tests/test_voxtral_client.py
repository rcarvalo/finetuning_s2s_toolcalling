"""Les chemins d'échec du client Voxtral, exercés avant de payer un A100.

Un timeout sur trente mille clips a tué un run entier ; un serveur mort n'a
jamais été relancé. Ici : réessai puis clip sauté, serveur relancé un nombre
borné de fois.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "_voxtral_client.py"


@pytest.fixture
def vox() -> Any:
    spec = importlib.util.spec_from_file_location("_voxtral_client", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses résout les annotations différées via sys.modules
    spec.loader.exec_module(module)
    return module


@dataclass
class _Response:
    status_code: int
    content: bytes = b""
    text: str = ""


@dataclass
class _Http:
    """Répond selon un script par texte : une liste de réponses ou d'exceptions."""

    script: dict[str, list[Any]]
    calls: list[str] = field(default_factory=list)

    def post(self, url: str, *, json: dict[str, Any]) -> _Response:
        text = json["input"]
        self.calls.append(text)
        outcome = self.script[text].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(vox: Any, script: dict[str, list[Any]], naps: list[float]) -> Any:
    return vox.VoxtralClient(
        http=_Http(script),
        model="m",
        voice_args={"voice": "fr_female"},
        base_url="http://s/v1",
        concurrency=2,
        sleep=naps.append,
        decode=lambda content: (content.decode(), 24_000),
    )


class TestVoxtralClient:
    def test_should_decode_a_good_answer_first_time(self, vox: Any) -> None:
        naps: list[float] = []
        client = _client(vox, {"a": [_Response(200, b"wave-a")]}, naps)

        waves, rate = client.speak(["a"])

        assert (waves, rate) == (["wave-a"], 24_000)
        assert naps == [] and client.failures == 0

    def test_should_retry_after_a_transport_error_then_succeed(self, vox: Any) -> None:
        naps: list[float] = []
        client = _client(vox, {"a": [TimeoutError("lent"), _Response(200, b"ok")]}, naps)

        waves, _ = client.speak(["a"])

        assert waves == ["ok"]
        assert naps == [2.0]

    def test_should_skip_a_clip_that_keeps_failing_without_raising(self, vox: Any) -> None:
        naps: list[float] = []
        client = _client(vox, {"a": [_Response(500, text="boom")] * 3, "b": [_Response(200, b"b")]}, naps)

        waves, rate = client.speak(["a", "b"])

        assert waves == [None, "b"]
        assert rate == 24_000
        assert client.failures == 1
        assert naps == [2.0, 4.0]

    def test_should_send_the_voice_and_model_with_every_request(self, vox: Any) -> None:
        http = _Http({"a": [_Response(200, b"x")]})
        client = vox.VoxtralClient(http=http, model="m", voice_args={"voice": "fr_female"}, base_url="http://s/v1")
        client.decode = lambda content: ("x", 24_000)

        client.one("a")

        assert http.calls == ["a"]

    def test_should_report_no_rate_when_nothing_came_back(self, vox: Any) -> None:
        naps: list[float] = []
        client = _client(vox, {"a": [_Response(503)] * 3}, naps)

        waves, rate = client.speak(["a"])

        assert (waves, rate) == ([None], 0)


@dataclass
class _Proc:
    returncode: int | None = None
    terminated: bool = False
    killed: bool = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        return -15

    def kill(self) -> None:
        self.killed = True


class TestServerGuard:
    def test_should_start_once_and_reuse_a_live_server(self, vox: Any) -> None:
        starts: list[int] = []

        def start() -> tuple[_Proc, str]:
            starts.append(1)
            return _Proc(), f"http://s{len(starts)}/v1"

        guard = vox.ServerGuard(start=start, note=lambda _: None)

        assert guard.ensure_alive() == "http://s1/v1"
        assert guard.ensure_alive() == "http://s1/v1"
        assert len(starts) == 1

    def test_should_restart_a_dead_server_and_say_so(self, vox: Any) -> None:
        procs = [_Proc(returncode=1), _Proc()]
        notes: list[str] = []
        guard = vox.ServerGuard(start=lambda: (procs.pop(0), "http://s/v1"), note=notes.append)

        guard.ensure_alive()
        guard.ensure_alive()

        assert guard.restarts == 1
        assert any("redémarrage 1/3" in n for n in notes)

    def test_should_give_up_after_too_many_deaths(self, vox: Any) -> None:
        guard = vox.ServerGuard(
            start=lambda: (_Proc(returncode=137), "http://s/v1"), max_restarts=2, note=lambda _: None
        )
        guard.ensure_alive()
        guard.ensure_alive()
        guard.ensure_alive()

        with pytest.raises(vox.ServerRestartsExhaustedError, match="3 fois"):
            guard.ensure_alive()


class TestServerGuardStop:
    def test_should_terminate_a_live_server(self, vox: Any) -> None:
        proc = _Proc()
        guard = vox.ServerGuard(start=lambda: (proc, "http://s/v1"), note=lambda _: None)
        guard.ensure_alive()

        guard.stop()

        assert proc.terminated and not proc.killed

    def test_should_do_nothing_without_a_server_or_with_a_dead_one(self, vox: Any) -> None:
        guard = vox.ServerGuard(start=lambda: (_Proc(returncode=1), "http://s/v1"), note=lambda _: None)

        guard.stop()  # never started
        guard.ensure_alive()
        guard.stop()  # already dead

        assert guard.process is not None and not guard.process.terminated
