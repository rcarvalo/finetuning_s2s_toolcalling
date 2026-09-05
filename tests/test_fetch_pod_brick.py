"""Le rapatriement depuis le pod : seulement ce que ni le Hub ni le disque n'ont."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from lfm2_audio.data_prep.corpus_layout import CorpusEntry

MODULE = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "fetch_pod_brick.py"


@pytest.fixture
def fetcher() -> Any:
    spec = importlib.util.spec_from_file_location("fetch_pod_brick", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entry(clip_id: str) -> CorpusEntry:
    return CorpusEntry(id=clip_id, audio=f"audio/{clip_id}.wav", text="x", lang="fr", duration_s=1.0)


def test_should_fetch_only_what_neither_the_hub_nor_the_disk_holds(fetcher: Any, tmp_path: Path) -> None:
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "local.wav").write_bytes(b"")
    entries = [_entry("on_hub_flat"), _entry("on_hub_sharded"), _entry("local"), _entry("missing")]
    on_hub = {"A/audio/on_hub_flat.wav", "A/audio/3f/on_hub_sharded.wav", "B/audio/missing.wav"}

    todo = fetcher.missing_downloads(entries, on_hub, tmp_path, "A")

    assert [e.id for e in todo] == ["missing"]
