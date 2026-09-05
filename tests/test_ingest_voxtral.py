"""L'ingestion d'un dossier Voxtral : ré-écoute, règle d'acceptation, reprise."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lfm2_audio.cli.data.ingest_voxtral import IngestConfig, VoxtralFolderIngester, build_parser


class EchoTranscriber:
    """Hears the text it is told to, except for stems listed as garbled."""

    def __init__(self, texts: dict[str, str], garbled: set[str]) -> None:
        self._texts, self._garbled = texts, garbled
        self.calls = 0

    def transcribe(self, audio: object, *, language: str | None = None) -> str:
        self.calls += 1
        stem = sorted(self._texts)[self.calls - 1]
        return "n'importe quoi de complètement différent" if stem in self._garbled else self._texts[stem]


@pytest.fixture
def source(tmp_path: Path) -> Path:
    folder = tmp_path / "audio"
    folder.mkdir()
    texts = {
        "sample_00000001": "Bonjour à tous.",
        "sample_00000002": "Il est dix-neuf heures.",
        "sample_00000003": "Sans wav.",
    }
    for stem, text in texts.items():
        (folder / f"{stem}.json").write_text(json.dumps({"text": text, "voice": "fr_female", "duration_sec": 0.5}))
        if stem != "sample_00000003":
            sf.write(folder / f"{stem}.wav", np.zeros(12000, dtype=np.float32), 24000, subtype="PCM_16")
    return folder


def test_should_list_only_the_pairs_with_audio(source: Path, tmp_path: Path) -> None:
    ingester = VoxtralFolderIngester(IngestConfig(source=source, out=tmp_path / "out"), EchoTranscriber({}, set()))

    assert [s.clip_id for s in ingester.samples()] == ["00000001", "00000002"]


def test_should_keep_what_says_its_text_and_log_the_rest(source: Path, tmp_path: Path) -> None:
    texts = {"sample_00000001": "Bonjour à tous.", "sample_00000002": "Il est 19h."}
    ingester = VoxtralFolderIngester(
        IngestConfig(source=source, out=tmp_path / "out"),
        EchoTranscriber(texts, {"sample_00000001"}),
        log=lambda _: None,
    )

    summary = ingester.run()

    assert (summary.clips, summary.new_clips, summary.dropped) == (1, 1, 1)
    manifest = [json.loads(line) for line in (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()]
    assert manifest[0]["id"] == "wiki_fr_00000002"
    assert manifest[0]["voxtral_wer"] == 0.0  # « 19h » entendu pour « dix-neuf heures » : forme parlée
    assert manifest[0]["speaker"] == "fr_female"
    assert (tmp_path / "out" / "audio" / "wiki_fr_00000002.wav").exists()
    dropped = json.loads((tmp_path / "out" / "dropped.jsonl").read_text().splitlines()[0])
    assert dropped["id"] == "wiki_fr_00000001"


def test_should_resume_without_re_listening(source: Path, tmp_path: Path) -> None:
    texts = {"sample_00000001": "Bonjour à tous.", "sample_00000002": "Il est dix-neuf heures."}
    config = IngestConfig(source=source, out=tmp_path / "out")
    VoxtralFolderIngester(config, EchoTranscriber(texts, {"sample_00000001"}), log=lambda _: None).run()
    second = EchoTranscriber(texts, set())

    summary = VoxtralFolderIngester(config, second, log=lambda _: None).run()

    assert second.calls == 0
    assert (summary.clips, summary.new_clips, summary.dropped) == (1, 0, 0)


def test_parser_should_default_to_the_long_form_brick_rule() -> None:
    args = build_parser().parse_args(["--source", "a", "--out", "b"])

    assert (args.max_wer, args.max_cer, args.device, args.id_prefix) == (0.15, 0.15, "cpu", "wiki_fr")


def test_retry_should_listen_again_only_to_the_rejected(source: Path, tmp_path: Path) -> None:
    texts = {"sample_00000001": "Bonjour à tous.", "sample_00000002": "Il est dix-neuf heures."}
    out = tmp_path / "out"
    VoxtralFolderIngester(
        IngestConfig(source=source, out=out), EchoTranscriber(texts, {"sample_00000001"}), log=lambda _: None
    ).run()
    stronger = EchoTranscriber({"sample_00000001": "Bonjour à tous."}, set())

    summary = VoxtralFolderIngester(
        IngestConfig(source=source, out=out, retry_rejected=True), stronger, log=lambda _: None
    ).run()

    assert stronger.calls == 1
    assert (summary.clips, summary.new_clips, summary.dropped) == (2, 1, 0)
