"""Le drain d'une brique : ce qui manque au Hub, à son chemin shardé, le reste intact."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from lfm2_audio.data_prep.corpus_layout import CorpusEntry, audio_relpath

MODULE = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "drain_brick.py"


@pytest.fixture
def drain() -> Any:
    spec = importlib.util.spec_from_file_location("drain_brick", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entry(clip_id: str, audio: str) -> CorpusEntry:
    return CorpusEntry(id=clip_id, audio=audio, text="x", lang="fr", duration_s=1.0)


def test_should_upload_only_what_the_hub_lacks_at_a_sharded_path(drain: Any, tmp_path: Path) -> None:
    (tmp_path / "audio").mkdir()
    for clip in ("old", "new"):
        (tmp_path / "audio" / f"{clip}.wav").write_bytes(b"")
    entries = [_entry("old", "audio/old.wav"), _entry("new", "audio/new.wav")]

    rewritten, uploads = drain.plan(entries, tmp_path, {"A/audio/old.wav"}, "A")

    assert [e.audio for e in rewritten] == ["audio/old.wav", audio_relpath("new")]
    assert [(u.local.name, u.path_in_repo) for u in uploads] == [("new.wav", f"A/{audio_relpath('new')}")]


def test_should_skip_a_clip_whose_sharded_copy_is_already_up(drain: Any, tmp_path: Path) -> None:
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "new.wav").write_bytes(b"")
    entries = [_entry("new", "audio/new.wav")]

    rewritten, uploads = drain.plan(entries, tmp_path, {f"A/{audio_relpath('new')}"}, "A")

    assert rewritten[0].audio == audio_relpath("new")
    assert uploads == []
