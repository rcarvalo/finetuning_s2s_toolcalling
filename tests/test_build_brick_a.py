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


class TestResolveSource:
    def test_should_wait_for_a_source_still_being_produced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path), BRICK_A_WAIT_SOURCES_MIN="2")
        import huggingface_hub

        landed = tmp_path / "landed.jsonl"
        attempts: list[int] = []

        def download(repo: str, path: str, repo_type: str) -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise FileNotFoundError(path)
            return str(landed)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
        naps: list[float] = []

        resolved = job._resolve_source(tmp_path / "corpus" / "C_dialogues" / "later.jsonl", sleep=naps.append)

        assert resolved == landed
        assert naps == [60, 60]

    def test_should_skip_a_missing_source_when_told_not_to_wait(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path))
        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub, "hf_hub_download", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("x"))
        )
        naps: list[float] = []

        assert job._resolve_source(tmp_path / "corpus" / "C_dialogues" / "absent.jsonl", sleep=naps.append) is None
        assert naps == []


class TestProbeGpu:
    def test_should_pass_when_the_subprocess_sees_a_gpu(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path))
        calls: list[list[str]] = []

        def run(cmd: list[str], check: bool) -> Any:
            calls.append(cmd)
            return type("R", (), {"returncode": 0})()

        job.probe_gpu(run=run)

        # torch is imported in the child only: the parent keeps its import table clean
        assert calls and "import torch" in calls[0][-1]

    def test_should_stop_before_installing_anything_without_a_gpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path))

        with pytest.raises(SystemExit, match="aucun GPU"):
            job.probe_gpu(run=lambda cmd, check: type("R", (), {"returncode": 1})())


class TestAcceptance:
    def test_should_keep_a_clip_under_the_word_threshold(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path))

        assert job.accepted(0.15, 0.5)

    def test_should_rescue_a_misheard_name_by_the_character_rate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path))

        assert job.accepted(0.22, 0.03)

    def test_should_refuse_a_clip_failing_both_rates(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path))

        assert not job.accepted(0.4, 0.3)

    def test_should_read_both_thresholds_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        job = _load(
            monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path), BRICK_A_MAX_WER="0.0", BRICK_A_MAX_CER="0.0"
        )

        assert not job.accepted(0.01, 0.01)
        assert job.MAX_ATTEMPTS == 2


class TestVerificationRates:
    def test_should_agree_when_only_the_number_spelling_differs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path))

        wer, cer = job.verification_rates(
            "L'accueil ferme à dix-neuf heures pile.", "L'accueil ferme à 19h pile.", "fr"
        )

        assert (wer, cer) == (0.0, 0.0)

    def test_should_still_measure_a_real_miss(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        job = _load(monkeypatch, tmp_path, BRICK_A_SOURCES=_source(tmp_path))

        wer, cer = job.verification_rates("Bon passage parmi nous !", "Bon passage.", "fr")

        assert wer == pytest.approx(0.5)
        assert cer > 0.10
