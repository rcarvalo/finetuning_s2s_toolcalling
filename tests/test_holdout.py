"""Tests of the cross-source hold-out filter."""

from __future__ import annotations

from pathlib import Path

import pytest

from lfm2_audio.data_prep.holdout import HoldoutFilter, normalise_transcript


@pytest.fixture
def benchmark_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "cv_fr_asr"
    directory.mkdir()
    (directory / "speakers.txt").write_text("spk_a\nspk_b\n", encoding="utf-8")
    (directory / "source_ids.txt").write_text("clip_1\nclip_2\n", encoding="utf-8")
    return directory


def test_should_load_both_exclusion_lists(benchmark_dir: Path) -> None:
    holdout = HoldoutFilter.from_benchmarks([benchmark_dir])

    assert holdout.speakers == {"spk_a", "spk_b"}
    assert holdout.source_ids == {"clip_1", "clip_2"}
    assert holdout.is_empty is False


def test_should_tolerate_a_benchmark_without_speaker_ids(tmp_path: Path) -> None:
    """FLEURS ships no speaker ids; that must not break the load."""
    directory = tmp_path / "fleurs_fr_asr"
    directory.mkdir()
    (directory / "source_ids.txt").write_text("fl_1\n", encoding="utf-8")

    holdout = HoldoutFilter.from_benchmarks([directory])

    assert holdout.speakers == set()
    assert holdout.source_ids == {"fl_1"}


def test_should_exclude_a_row_by_speaker(benchmark_dir: Path) -> None:
    """The point of the whole module: the SAME speaker reaching training
    through a different source than the one the benchmark was cut from."""
    holdout = HoldoutFilter.from_benchmarks([benchmark_dir])

    assert holdout.excludes({"speaker": "spk_a", "id": "other_source_id"}) is True
    assert holdout.stats.by_speaker == 1


def test_should_exclude_a_row_by_source_id(benchmark_dir: Path) -> None:
    holdout = HoldoutFilter.from_benchmarks([benchmark_dir])

    assert holdout.excludes({"speaker": "unknown", "id": "clip_2"}) is True
    assert holdout.stats.by_source_id == 1


def test_should_keep_an_unrelated_row(benchmark_dir: Path) -> None:
    holdout = HoldoutFilter.from_benchmarks([benchmark_dir])

    assert holdout.excludes({"speaker": "spk_z", "id": "clip_9"}) is False
    assert holdout.stats.kept == 1


def test_should_exclude_by_transcript_when_given_one(benchmark_dir: Path) -> None:
    """Re-encoded through another pipeline, a clip keeps neither id nor
    speaker — the sentence is what is left to match on."""
    holdout = HoldoutFilter.from_benchmarks([benchmark_dir], transcripts=["Il a été président du Sénat."])

    assert holdout.excludes({"speaker": "spk_z", "id": "x", "text": "il a ete president du senat"}) is True
    assert holdout.stats.by_transcript == 1


def test_keep_should_filter_a_whole_stream(benchmark_dir: Path) -> None:
    holdout = HoldoutFilter.from_benchmarks([benchmark_dir])
    rows = [
        {"speaker": "spk_a", "id": "r1"},
        {"speaker": "spk_z", "id": "r2"},
        {"speaker": "spk_y", "id": "clip_1"},
        {"speaker": "spk_x", "id": "r4"},
    ]

    kept = holdout.keep(rows)

    assert [row["id"] for row in kept] == ["r2", "r4"]
    assert holdout.stats.dropped == 2
    assert "2 exclus" in holdout.stats.summary()


def test_empty_filter_should_announce_itself(tmp_path: Path) -> None:
    """An empty hold-out silently keeping everything is the failure this flags."""
    empty = tmp_path / "nothing"
    empty.mkdir()

    holdout = HoldoutFilter.from_benchmarks([empty])

    assert holdout.is_empty is True


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Il a été président.", "il a ete president"),
        ("  Bonjour,   TOUS !  ", "bonjour tous"),
        ("L'été — déjà ?", "l ete deja"),
    ],
)
def test_normalise_transcript_should_fold_accents_case_and_punctuation(left: str, right: str) -> None:
    assert normalise_transcript(left) == right
