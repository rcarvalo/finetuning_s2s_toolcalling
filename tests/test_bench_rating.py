"""Tests for the listening bench: rating value object and its store."""

from __future__ import annotations

import json

import pytest

from lfm2_audio.bench.rating import SCALE_MAX, SCALE_MIN, Rating
from lfm2_audio.bench.store import RatingStore


def _rating(case: str = "c1", version: str = "v1", **overrides) -> Rating:
    values = {"intelligibility": 4, "naturalness": 3, "overall": 4}
    values.update(overrides)
    return Rating.create(case, version, **values)


# --------------------------------------------------------------------------- #
# Rating
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("axis", ["intelligibility", "naturalness", "overall"])
@pytest.mark.parametrize("value", [0, 6, -1])
def test_should_reject_scores_outside_the_scale(axis, value):
    with pytest.raises(ValueError, match=axis):
        _rating(**{axis: value})


@pytest.mark.parametrize("value", [SCALE_MIN, 3, SCALE_MAX])
def test_should_accept_the_whole_scale(value):
    assert _rating(overall=value).overall == value


def test_should_stamp_the_time_of_judgement():
    assert _rating().rated_at != ""


def test_mean_should_average_the_three_axes():
    assert _rating(intelligibility=3, naturalness=3, overall=3).mean_score == 3.0


def test_should_roundtrip_through_a_dict():
    original = _rating(notes="clipped at the end", derailed=True)

    restored = Rating.from_dict(original.as_dict())

    assert restored == original


def test_derailed_defaults_to_false():
    assert _rating().derailed is False


# --------------------------------------------------------------------------- #
# RatingStore
# --------------------------------------------------------------------------- #


def test_should_start_empty(tmp_path):
    assert RatingStore(tmp_path / "none.jsonl").load() == []


def test_should_append_without_rewriting(tmp_path):
    store = RatingStore(tmp_path / "nested" / "r.jsonl")

    store.append(_rating("c1"))
    store.append(_rating("c2"))

    assert [r.case_id for r in store.load()] == ["c1", "c2"]


def test_should_skip_a_corrupt_line_instead_of_failing(tmp_path):
    """A half-written line must not cost a whole session of judgements."""
    path = tmp_path / "r.jsonl"
    store = RatingStore(path)
    store.append(_rating("c1"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    store.append(_rating("c2"))

    assert [r.case_id for r in store.load()] == ["c1", "c2"]


def test_rated_cases_should_be_scoped_to_one_version(tmp_path):
    """Ratings from another checkpoint must not mark this one as done."""
    store = RatingStore(tmp_path / "r.jsonl")
    store.append(_rating("c1", version="vanilla"))
    store.append(_rating("c2", version="finetuned"))

    assert store.rated_cases("vanilla") == {"c1"}
    assert store.versions() == ["finetuned", "vanilla"]


def test_summary_should_count_derailments_apart_from_scores(tmp_path):
    """A broken generation is not a low score; averaging them hides the failure."""
    store = RatingStore(tmp_path / "r.jsonl")
    store.append(_rating("c1", overall=5, intelligibility=5, naturalness=5))
    store.append(_rating("c2", overall=1, intelligibility=1, naturalness=1, derailed=True))

    summary = store.summary("v1")

    assert summary["rated"] == 2
    assert summary["derailed"] == 1
    assert summary["overall"] == 5.0  # the derailed clip is excluded from the mean


def test_summary_of_an_unknown_version_is_empty(tmp_path):
    assert RatingStore(tmp_path / "r.jsonl").summary("nope") == {"rated": 0}


def test_by_case_should_group_versions_together(tmp_path):
    store = RatingStore(tmp_path / "r.jsonl")
    store.append(_rating("c1", version="vanilla"))
    store.append(_rating("c1", version="finetuned"))

    grouped = store.by_case()

    assert sorted(r.version for r in grouped["c1"]) == ["finetuned", "vanilla"]


def test_records_should_be_valid_json_lines(tmp_path):
    path = tmp_path / "r.jsonl"
    RatingStore(path).append(_rating())

    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["case_id"] == "c1"
