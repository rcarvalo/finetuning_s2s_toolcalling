"""Corpus curation (`lfm2_audio.data_prep.curation`) — dedup and leakage guard."""

from __future__ import annotations

import json
from typing import Any

from lfm2_audio.data_prep.curation import (
    CurationReport,
    curate,
    first_user_text,
    load_jsonl,
    normalize_utterance,
    write_jsonl,
)


def _dialogue(idx: str, text: str) -> dict[str, Any]:
    return {"id": idx, "tools": [], "turns": [{"role": "user", "text": text}]}


class TestNormalizeUtterance:
    def test_should_ignore_case_punctuation_and_spacing(self) -> None:
        assert normalize_utterance("Find  my ORDERS, please!") == normalize_utterance("find my orders please")

    def test_should_match_a_contraction_with_its_spelled_out_form(self) -> None:
        assert normalize_utterance("What's the weather?") == normalize_utterance("whats the weather")

    def test_should_fold_accents(self) -> None:
        assert normalize_utterance("Café") == normalize_utterance("cafe")

    def test_should_return_empty_for_punctuation_only(self) -> None:
        assert normalize_utterance("?!...") == ""


class TestFirstUserText:
    def test_should_return_the_first_user_turn(self) -> None:
        dialogue = {"turns": [{"role": "system", "text": "s"}, {"role": "user", "text": "hi"}]}

        assert first_user_text(dialogue) == "hi"

    def test_should_return_empty_when_no_user_turn(self) -> None:
        assert first_user_text({"turns": [{"role": "assistant", "text": "hi"}]}) == ""


class TestCurate:
    def test_should_keep_every_distinct_dialogue(self) -> None:
        kept, report = curate({"a": [_dialogue("1", "hello"), _dialogue("2", "goodbye")]})

        assert len(kept) == 2
        assert report.kept == 2
        assert report.dropped == 0

    def test_should_drop_duplicates_across_sources_keeping_the_first(self) -> None:
        sources = {"trusted": [_dialogue("1", "What's the weather?")], "extra": [_dialogue("9", "whats the weather")]}

        kept, report = curate(sources)

        assert [d["id"] for d in kept] == ["1"]
        assert report.duplicates == 1
        assert report.per_source == {"trusted": 1, "extra": 0}

    def test_should_drop_dialogues_present_in_the_held_out_split(self) -> None:
        kept, report = curate(
            {"train": [_dialogue("1", "Find my orders"), _dialogue("2", "Play music")]},
            held_out=[_dialogue("t1", "find my orders")],
        )

        assert [d["id"] for d in kept] == ["2"]
        assert report.leaked == 1

    def test_should_count_dialogues_without_a_user_turn_as_empty(self) -> None:
        kept, report = curate({"a": [{"id": "x", "turns": [{"role": "assistant", "text": "hi"}]}]})

        assert kept == []
        assert report.empty == 1

    def test_should_ignore_blank_held_out_entries(self) -> None:
        """An empty held-out key must not swallow every dialogue."""
        kept, _ = curate({"a": [_dialogue("1", "hello")]}, held_out=[{"turns": []}])

        assert len(kept) == 1

    def test_should_preserve_source_order(self) -> None:
        sources = {"first": [_dialogue("1", "a")], "second": [_dialogue("2", "b")]}

        kept, _ = curate(sources)

        assert [d["id"] for d in kept] == ["1", "2"]


class TestReport:
    def test_should_summarize_what_was_dropped(self) -> None:
        report = CurationReport(kept=10, duplicates=3, leaked=2, empty=1, per_source={"a": 10})

        summary = report.summary()

        assert "kept 10" in summary
        assert "dropped 6" in summary
        assert "a=10" in summary


class TestJsonlRoundTrip:
    def test_should_write_then_reload_dialogues(self, tmp_path) -> None:
        path = tmp_path / "out" / "corpus.jsonl"

        write_jsonl([_dialogue("1", "héllo")], path)

        assert [d["id"] for d in load_jsonl(path)] == ["1"]

    def test_should_skip_blank_lines_when_loading(self, tmp_path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text(json.dumps(_dialogue("1", "a")) + "\n\n", encoding="utf-8")

        assert len(list(load_jsonl(path))) == 1
