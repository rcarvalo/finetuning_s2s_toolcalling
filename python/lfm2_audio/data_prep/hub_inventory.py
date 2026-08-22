"""Inventory of a Hub author's datasets — the raw material of a training run.

Weekend step 3: before picking training data, list what already exists under
``Rcarvalo/`` (~45 audio datasets), with the tags and sizes needed to decide
what can improve WER / DNSMOS / NISQA / tool-calling scores.

``huggingface_hub`` objects are duck-typed on purpose: tests inject a fake API
and the module never needs network access to be imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class _HubApi(Protocol):
    """The two calls we need from ``HfApi`` — injectable in tests."""

    def list_datasets(self, *, author: str) -> Any: ...  # noqa: ANN401 — untyped upstream iterator
    def dataset_info(self, repo_id: str, *, files_metadata: bool) -> Any: ...  # noqa: ANN401


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    """One dataset repo, summarized for the inventory table."""

    repo_id: str
    private: bool
    size_mb: float
    files: int
    tags: tuple[str, ...]
    last_modified: str

    @property
    def visibility(self) -> str:
        return "private" if self.private else "public"


def scan_author(api: _HubApi, author: str) -> list[DatasetEntry]:
    """List every dataset of ``author``, newest first, with file sizes."""
    entries: list[DatasetEntry] = []
    for item in api.list_datasets(author=author):
        info = api.dataset_info(item.id, files_metadata=True)
        siblings = getattr(info, "siblings", None) or []
        size = sum((s.size or 0) for s in siblings)
        entries.append(
            DatasetEntry(
                repo_id=info.id,
                private=bool(getattr(info, "private", False)),
                size_mb=round(size / 1_000_000, 1),
                files=len(siblings),
                tags=tuple(getattr(info, "tags", None) or ()),
                last_modified=str(getattr(info, "last_modified", "") or "")[:10],
            )
        )
    entries.sort(key=lambda e: e.last_modified, reverse=True)
    return entries


def to_markdown(entries: list[DatasetEntry], author: str) -> str:
    """Render the inventory as a Markdown document (checked into ``docs/``)."""
    lines = [
        f"# Dataset inventory — {author}",
        "",
        f"{len(entries)} dataset(s). Sizes are per-repo file totals.",
        "",
        "| Repo | Visibility | Size (MB) | Files | Updated | Tags |",
        "|---|---|---:|---:|---|---|",
    ]
    for e in entries:
        tags = ", ".join(t for t in e.tags if not t.startswith("region:"))[:80]
        lines.append(f"| {e.repo_id} | {e.visibility} | {e.size_mb} | {e.files} | {e.last_modified} | {tags} |")
    lines.append("")
    return "\n".join(lines)


def write_inventory(api: _HubApi, author: str, out: str | Path) -> list[DatasetEntry]:
    """Scan and write the Markdown inventory; returns the entries for callers."""
    entries = scan_author(api, author)
    Path(out).write_text(to_markdown(entries, author), encoding="utf-8")
    return entries
