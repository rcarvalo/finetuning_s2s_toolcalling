"""Le pré-vol Gemini doit tenir son client le temps de l'appel.

Créé en expression temporaire, le client google-genai est fermé avant que la
requête parte, et le pré-vol rejette alors une clé VALIDE — mesuré en local le
28/08 sur la clé qui fait tourner le juge.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, ClassVar

import pytest

PREFLIGHT = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "_gemini_preflight.py"


@pytest.fixture
def preflight() -> Any:
    spec = importlib.util.spec_from_file_location("_gemini_preflight", PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeModels:
    def __init__(self, owner: _FakeClient) -> None:
        self._owner = owner

    def generate_content(self, model: str, contents: str) -> None:
        if self._owner.closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        self._owner.calls.append((model, contents))


class _FakeClient:
    instances: ClassVar[list[_FakeClient]] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.calls: list[tuple[str, str]] = []
        self.closed = False
        self.models = _FakeModels(self)
        _FakeClient.instances.append(self)


def test_should_make_one_call_with_the_key(monkeypatch: pytest.MonkeyPatch, preflight: Any) -> None:
    from google import genai

    monkeypatch.setattr(genai, "Client", _FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.abc")  # pragma: allowlist secret

    preflight.preflight()

    assert _FakeClient.instances[-1].api_key == "AQ.abc"  # pragma: allowlist secret
    assert len(_FakeClient.instances[-1].calls) == 1


def test_should_exit_on_an_empty_key(monkeypatch: pytest.MonkeyPatch, preflight: Any) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "  ")

    with pytest.raises(SystemExit, match="manquant"):
        preflight.preflight()


def test_should_exit_with_the_key_shape_when_gemini_refuses(monkeypatch: pytest.MonkeyPatch, preflight: Any) -> None:
    from google import genai

    class _Refusing(_FakeClient):
        def __init__(self, api_key: str) -> None:
            super().__init__(api_key)
            self.closed = True  # every call fails, whatever the reason

    monkeypatch.setattr(genai, "Client", _Refusing)
    monkeypatch.setenv("GEMINI_API_KEY", "ya29.token")  # pragma: allowlist secret

    with pytest.raises(SystemExit, match="'ya2', 10 caractères"):
        preflight.preflight()
