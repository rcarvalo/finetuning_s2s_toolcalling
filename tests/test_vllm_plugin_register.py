"""The plugin must decline an unknown host, never break it.

Regression for a costly failure: this repo declares a
``vllm_omni.general_plugins`` entry point, so vLLM-Omni loads ``register()`` in
every process it starts — including ones serving a completely different model.
Against vllm-omni 0.26, where ``register_pipeline`` moved, the ImportError
surfaced as "Orchestrator initialization failed" and made Voxtral unlaunchable
on any machine where this repo was installed. Six runs were spent blaming pins,
CUDA and the GPU.
"""

from __future__ import annotations

import sys
import types

import pytest

from lfm2_audio import vllm_plugin


@pytest.fixture
def host_without_register_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vLLM-Omni whose ``stage_config`` lacks the 0.22 symbol."""
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    executor = types.ModuleType("vllm.model_executor.models")
    executor.ModelRegistry = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.model_executor", types.ModuleType("vllm.model_executor"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", executor)

    monkeypatch.setitem(sys.modules, "vllm_omni", types.ModuleType("vllm_omni"))
    monkeypatch.setitem(sys.modules, "vllm_omni.config", types.ModuleType("vllm_omni.config"))
    monkeypatch.setitem(sys.modules, "vllm_omni.config.stage_config", types.ModuleType("vllm_omni.config.stage_config"))


def test_should_return_quietly_when_host_lacks_register_pipeline(
    host_without_register_pipeline: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        vllm_plugin.register()

    assert "désactivé" in caplog.text
    assert "register_pipeline" in caplog.text


def test_should_not_import_heavy_modules_when_declining(host_without_register_pipeline: None) -> None:
    """Declining must not drag liquid_audio in: the host process has its own model to load."""
    vllm_plugin.register()

    assert "lfm2_audio.vllm_plugin.omni_model" not in sys.modules
