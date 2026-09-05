"""Le job de la brique A : familles filtrées, manifeste du Hub conservé à la reprise."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

JOB = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "build_brick_a.py"


def _load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **env: str) -> Any:
    monkeypatch.setenv("LFM2_ROOT", str(tmp_path))
    monkeypatch.setenv("LFM2_OUT", str(tmp_path / "out"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location("build_brick_a", JOB)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(tmp_path: Path) -> str:
    rows = [
        {
            "id": "c_deep_s00_0001",
            "meta": {"lang": "fr", "kind": "fr_deep"},
            "turns": [{"role": "user", "text": "Bonjour"}, {"role": "assistant", "text": "Bonjour, je vous écoute."}],
        },
        {
            "id": "c_en_s00_0001",
            "meta": {"lang": "en", "kind": "en"},
            "turns": [{"role": "user", "text": "Hi"}, {"role": "assistant", "text": "Hello, how can I help?"}],
        },
        {
            "id": "tcfr_000001",
            "meta": {"lang": "fr"},
            "turns": [{"role": "user", "text": "Météo ?"}, {"role": "assistant", "text": "Il fait beau."}],
        },
    ]
    path = tmp_path / "corpus" / "C_dialogues" / "dialogues_v2.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return "corpus/C_dialogues/dialogues_v2.jsonl"


class TestTurnsToSpeak:
    def test_should_take_every_assistant_turn_by_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path))

        ids = [cid for cid, _, _ in job.turns_to_speak(None)]

        assert ids == ["c_deep_s00_0001_t1", "c_en_s00_0001_t1", "tcfr_000001_t1"]

    def test_should_leave_out_the_skipped_families(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path), BRICK_A_SKIP_KINDS="en")

        ids = [cid for cid, _, _ in job.turns_to_speak(None)]

        assert ids == ["c_deep_s00_0001_t1", "tcfr_000001_t1"]

    def test_should_keep_only_the_named_families(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path), BRICK_A_KINDS="en")

        items = job.turns_to_speak(None)

        assert [(cid, lang) for cid, _, lang in items] == [("c_en_s00_0001_t1", "en")]


class TestMergeExisting:
    def test_should_start_from_the_hub_manifest(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path))
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"id": "old_t1", "audio": "audio/old_t1.wav", "text": "x", "lang": "fr", "duration_s": 1.0})
            + "\n"
        )

        kept = job.merge_existing(manifest)

        assert [entry.id for entry in kept] == ["old_t1"]

    def test_should_start_empty_without_a_manifest(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path))

        assert job.merge_existing(None) == []
        assert job.merge_existing(tmp_path / "absent.jsonl") == []
