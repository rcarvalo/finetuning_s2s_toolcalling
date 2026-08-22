"""Hub dataset inventory (`lfm2_audio.data_prep.hub_inventory`) — fake API, no network."""

from __future__ import annotations

from dataclasses import dataclass, field

from lfm2_audio.data_prep.hub_inventory import scan_author, to_markdown, write_inventory


@dataclass
class FakeSibling:
    size: int | None


@dataclass
class FakeInfo:
    id: str
    private: bool = True
    siblings: list[FakeSibling] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    last_modified: str = "2026-08-01T00:00:00"


class FakeApi:
    def __init__(self, infos: list[FakeInfo]) -> None:
        self._infos = {i.id: i for i in infos}

    def list_datasets(self, *, author: str):
        assert author == "Rcarvalo"
        return list(self._infos.values())

    def dataset_info(self, repo_id: str, *, files_metadata: bool):
        assert files_metadata
        return self._infos[repo_id]


def _api() -> FakeApi:
    return FakeApi(
        [
            FakeInfo(
                "Rcarvalo/old-corpus",
                private=False,
                siblings=[FakeSibling(2_000_000), FakeSibling(None)],
                tags=["audio", "region:us"],
                last_modified="2026-01-15T09:00:00",
            ),
            FakeInfo(
                "Rcarvalo/tc-en-audio",
                siblings=[FakeSibling(5_500_000)],
                tags=["audio", "en"],
                last_modified="2026-08-10T12:00:00",
            ),
        ]
    )


def test_should_scan_sizes_and_sort_newest_first() -> None:
    entries = scan_author(_api(), "Rcarvalo")

    assert [e.repo_id for e in entries] == ["Rcarvalo/tc-en-audio", "Rcarvalo/old-corpus"]
    assert entries[1].size_mb == 2.0  # None-sized sibling counts as 0
    assert entries[1].files == 2
    assert entries[0].last_modified == "2026-08-10"


def test_should_expose_visibility_label() -> None:
    entries = scan_author(_api(), "Rcarvalo")

    assert entries[0].visibility == "private"
    assert entries[1].visibility == "public"


def test_should_render_markdown_without_region_tags() -> None:
    markdown = to_markdown(scan_author(_api(), "Rcarvalo"), "Rcarvalo")

    assert "| Rcarvalo/tc-en-audio | private | 5.5 | 1 | 2026-08-10 | audio, en |" in markdown
    assert "region:us" not in markdown


def test_should_write_inventory_file(tmp_path) -> None:
    out = tmp_path / "inventory.md"

    entries = write_inventory(_api(), "Rcarvalo", out)

    assert len(entries) == 2
    assert out.read_text(encoding="utf-8").startswith("# Dataset inventory — Rcarvalo")


def test_should_handle_author_without_datasets() -> None:
    entries = scan_author(FakeApi([]), "Rcarvalo")

    assert entries == []
    assert "0 dataset(s)" in to_markdown(entries, "Rcarvalo")
