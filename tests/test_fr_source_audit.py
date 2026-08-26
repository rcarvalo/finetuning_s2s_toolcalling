"""Tests of the FR source audit aggregation (no network, no VERSA)."""

from __future__ import annotations

from lfm2_audio.data_prep.fr_source_audit import ClipAudit, SourceAudit, audit_markdown


def _audit_with(clips: list[ClipAudit]) -> SourceAudit:
    audit = SourceAudit(name="src", register="lu")
    for clip in clips:
        audit.add(clip)
    return audit


def test_should_compute_medians_ignoring_missing_values() -> None:
    audit = _audit_with(
        [
            ClipAudit(sample_id="a", utmos=4.0, duration_s=3.0),
            ClipAudit(sample_id="b", utmos=2.0, duration_s=5.0),
            ClipAudit(sample_id="c", utmos=None, duration_s=None),
        ]
    )

    assert audit.median("utmos") == 3.0
    assert audit.median("duration_s") == 4.0


def test_median_should_be_none_without_any_measurement() -> None:
    audit = _audit_with([ClipAudit(sample_id="a")])

    assert audit.median("nisqa") is None


def test_should_count_distinct_named_speakers() -> None:
    audit = _audit_with(
        [
            ClipAudit(sample_id="a", speaker="x"),
            ClipAudit(sample_id="b", speaker="x"),
            ClipAudit(sample_id="c", speaker="y"),
            ClipAudit(sample_id="d", speaker=""),
        ]
    )

    assert audit.speaker_count == 2


def test_duration_span_should_need_ten_clips() -> None:
    audit = _audit_with([ClipAudit(sample_id=str(i), duration_s=float(i)) for i in range(10)])

    assert audit.duration_p10_p90() == (1.0, 9.0)
    assert _audit_with([ClipAudit(sample_id="a", duration_s=1.0)]).duration_p10_p90() is None


def test_markdown_should_render_one_row_per_source_with_dashes_for_missing() -> None:
    measured = _audit_with([ClipAudit(sample_id="a", utmos=4.1, dnsmos=3.2, duration_s=3.0)])
    metadata_only = SourceAudit(name="codes", register="spontané", metadata_only=True)
    metadata_only.add(ClipAudit(sample_id="b", duration_s=7.0))

    table = audit_markdown([measured, metadata_only])

    lines = table.splitlines()
    assert len(lines) == 4  # header + separator + 2 rows
    assert "| src |" in lines[2] and "4.10" in lines[2]
    assert "codes (métadonnées seules)" in lines[3] and "—" in lines[3]
