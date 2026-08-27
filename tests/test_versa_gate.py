"""Tests of the VERSA gate aggregation."""

from __future__ import annotations

from pathlib import Path

from lfm2_audio.evaluation.eval_log_audio import LoggedReply
from lfm2_audio.evaluation.versa_gate import build_report


def _reply(sample_id: str, text: str) -> LoggedReply:
    return LoggedReply(
        sample_id=sample_id,
        text=text,
        wav_path=Path(f"/tmp/{sample_id}.wav"),
        seconds=1.0,
        question_language="fr",
    )


FRENCH = "Bonjour je suis ravi de vous aider avec cette question aujourd'hui."
ENGLISH = "Sure the weather is sunny today and it will stay warm this week."


def test_should_split_metrics_by_the_language_actually_spoken() -> None:
    """Both replies answer FRENCH questions; one answers in English. Pooling
    them would hide the very gap the bilingual gates exist to catch."""
    replies = [_reply("a", FRENCH), _reply("b", ENGLISH)]
    scores = {
        "a": {"utmos": 3.5, "dns_overall": 3.0, "nisqa_mos_pred": 3.8},
        "b": {"utmos": 4.5, "dns_overall": 3.4, "nisqa_mos_pred": 4.2},
    }

    report = build_report(replies, scores)

    assert report.by_language["fr"]["utmos"].median == 3.5
    assert report.by_language["en"]["utmos"].median == 4.5
    assert report.languages_seen == {"fr": 1, "en": 1}


def test_pooled_should_cover_every_reply() -> None:
    replies = [_reply("a", FRENCH), _reply("b", ENGLISH)]
    scores = {"a": {"utmos": 3.0}, "b": {"utmos": 4.0}}

    report = build_report(replies, scores)

    assert report.pooled["utmos"].n == 2
    assert report.pooled["utmos"].median == 3.5


def test_should_ignore_a_reply_versa_did_not_score() -> None:
    replies = [_reply("a", FRENCH), _reply("missing", FRENCH)]

    report = build_report(replies, {"a": {"utmos": 3.0}})

    assert report.pooled["utmos"].n == 1
    assert report.languages_seen == {"fr": 1}


def test_unclassifiable_speech_should_get_its_own_bucket() -> None:
    """Never silently folded into a language: a degenerate reply would drag
    that language's median without belonging to it."""
    replies = [_reply("a", "1969")]

    report = build_report(replies, {"a": {"utmos": 2.0}})

    assert report.languages_seen == {"?": 1}


def test_markdown_should_render_a_row_per_language_and_a_total() -> None:
    replies = [_reply("a", FRENCH), _reply("b", ENGLISH)]
    scores = {"a": {"utmos": 3.5}, "b": {"utmos": 4.5}}

    table = build_report(replies, scores).markdown()

    lines = table.splitlines()
    assert len(lines) == 5  # header + separator + en + fr + total
    assert "**toutes**" in lines[-1]
    assert "—" in lines[2]  # DNSMOS absent des scores fournis
