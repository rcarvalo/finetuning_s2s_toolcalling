"""Tests of the FR corpus layout and manifest contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from lfm2_audio.data_prep.corpus_layout import (
    BRICKS,
    BRICKS_BY_KEY,
    CorpusEntry,
    CorpusError,
    brick_readme,
    read_manifest,
    write_manifest,
)


def _entry(**overrides: object) -> CorpusEntry:
    base = {
        "id": "a_0001",
        "audio": "audio/a_0001.wav",
        "text": "Bonjour, comment puis-je vous aider ?",
        "lang": "fr",
        "duration_s": 2.5,
    }
    return CorpusEntry(**{**base, **overrides})  # type: ignore[arg-type]


def test_the_four_bricks_are_declared_with_distinct_folders() -> None:
    assert [b.key for b in BRICKS] == ["A", "B", "C", "D", "E"]
    assert len({b.folder for b in BRICKS}) == 5
    assert BRICKS_BY_KEY["A"].folder == "A_assistant_speech"


def test_should_round_trip_an_entry_through_the_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"

    written = write_manifest([_entry(voxtral_wer=0.02, speaker="qwen_fr_1")], path)
    back = list(read_manifest(path))

    assert written == 1
    assert back[0].id == "a_0001"
    assert back[0].voxtral_wer == 0.02
    assert back[0].speaker == "qwen_fr_1"


def test_unknown_columns_should_survive_in_extra(tmp_path: Path) -> None:
    """A brick may carry its own provenance fields; losing them on read would
    quietly erase why a clip was kept."""
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"id": "b_1", "audio": "audio/b_1.wav", "text": "bonjour tout le monde",'
        ' "lang": "fr", "duration_s": 1.5, "distillmos": 3.9}\n',
        encoding="utf-8",
    )

    entry = next(read_manifest(path))

    assert entry.extra == {"distillmos": 3.9}


@pytest.mark.parametrize(
    ("field", "value"),
    [("lang", "de"), ("role", "system"), ("duration_s", 0.0), ("text", "   ")],
)
def test_should_reject_an_invalid_entry(field: str, value: object) -> None:
    with pytest.raises(CorpusError):
        _entry(**{field: value}).validate()


def test_write_should_reject_before_writing_anything(tmp_path: Path) -> None:
    """Validation at the boundary: once a bad row is in the corpus it is
    silently trained on."""
    path = tmp_path / "manifest.jsonl"

    with pytest.raises(CorpusError):
        write_manifest([_entry(), _entry(id="bad", lang="de")], path)


def test_empty_optional_fields_should_not_be_written(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest([_entry()], path)

    payload = path.read_text(encoding="utf-8")

    assert "speaker" not in payload
    assert "voxtral_wer" not in payload


def test_readme_should_state_the_brick_contract() -> None:
    readme = brick_readme(BRICKS_BY_KEY["B"], entries=1200, hours=3.5)

    assert "B_user_speech" in readme
    assert "diversité maximale" in readme
    assert "1200 clips" in readme
    assert "voxtral_wer" in readme
