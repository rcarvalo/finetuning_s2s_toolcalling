"""Regression: the liquid backend must hand a `Path` to liquid-audio.

`liquid_audio.utils.get_model_dir` overloads its argument by type — a `str` is
downloaded as a Hub repo id, a `Path` is read as a local directory. Passing
`str(resolved_path)` made every local checkpoint fail with HFValidationError.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from lfm2_audio.ds.checkpoint import Layout, ResolvedCheckpoint


class _Recorder:
    """Stands in for LFM2AudioModel / LFM2AudioProcessor."""

    def __init__(self) -> None:
        self.seen: list[object] = []

    def from_pretrained(self, source: object, **_: object) -> _Recorder:
        self.seen.append(source)
        return self

    def eval(self) -> _Recorder:
        return self


@pytest.fixture
def liquid_backend(monkeypatch: pytest.MonkeyPatch) -> tuple[type, _Recorder, _Recorder]:
    """Import the backend with `liquid_audio` and `torch` stubbed out."""
    model, processor = _Recorder(), _Recorder()

    liquid_audio = types.ModuleType("liquid_audio")
    liquid_audio.ChatState = object  # type: ignore[attr-defined]
    liquid_audio.LFM2AudioModel = model  # type: ignore[attr-defined]
    liquid_audio.LFM2AudioProcessor = processor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "liquid_audio", liquid_audio)

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    monkeypatch.delitem(sys.modules, "lfm2_audio.serving.backends.liquid", raising=False)
    from lfm2_audio.serving.backends.liquid import LiquidAudioBackend

    # The constructor warms up the detokenizer: only `_build`'s argument matters here.
    monkeypatch.setattr(LiquidAudioBackend, "__init__", lambda self, *a, **k: None)
    return LiquidAudioBackend, model, processor


def test_should_pass_a_path_not_a_string_to_liquid_audio(liquid_backend, tmp_path: Path) -> None:
    backend, model, processor = liquid_backend
    checkpoint = ResolvedCheckpoint(path=tmp_path / "snapshot", layout=Layout.LIQUID)

    backend._build(checkpoint, system="", engine=None, generation=None)

    assert model.seen == [tmp_path / "snapshot"], "a str would be downloaded as a repo id"
    assert processor.seen == [tmp_path / "snapshot"]
    assert all(isinstance(seen, Path) for seen in model.seen + processor.seen)
