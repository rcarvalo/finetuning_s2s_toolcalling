"""Push what a pod holds of a brick to the Hub, in small sharded commits.

The streaming pusher sends everything not yet pushed in ONE commit: once a
commit fails (the Hub refused the 10 001st file of ``audio/`` on 05/09), the
backlog only grows and every retry fails slower. This drains it: every clip of
the local manifest missing from the Hub goes up in chunks of ``--chunk``
files, moved to its sharded path, then the rewritten manifest and the logs.
Re-runnable: what is already there is skipped. ::

    python infra/jobs/drain_brick.py --out /workspace/out/A_assistant_speech --brick A_assistant_speech
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from lfm2_audio.data_prep.corpus_layout import CorpusEntry, audio_relpath, read_manifest, write_manifest  # noqa: E402


@dataclass(frozen=True)
class Upload:
    local: Path
    path_in_repo: str


def plan(entries: list[CorpusEntry], out: Path, on_hub: set[str], brick: str) -> tuple[list[CorpusEntry], list[Upload]]:
    """Entries with their final (sharded) audio path, and the files the Hub still lacks."""
    rewritten, uploads = [], []
    for entry in entries:
        flat = f"{brick}/{entry.audio}"
        if flat in on_hub:  # already there, flat or sharded: leave it alone
            rewritten.append(entry)
            continue
        target = audio_relpath(entry.id) if entry.audio.count("/") == 1 else entry.audio
        rewritten.append(CorpusEntry(**{**entry.__dict__, "audio": target}))
        local = out / entry.audio
        if local.exists() and f"{brick}/{target}" not in on_hub:
            uploads.append(Upload(local=local, path_in_repo=f"{brick}/{target}"))
    return rewritten, uploads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--brick", required=True)
    parser.add_argument("--repo", default="Rcarvalo/lfm25-fr-corpus-v1")
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    on_hub = set(api.list_repo_files(args.repo, repo_type="dataset"))
    entries = list(read_manifest(args.out / "manifest.jsonl"))
    rewritten, uploads = plan(entries, args.out, on_hub, args.brick)
    print(f"{len(entries)} entrées locales, {len(uploads)} fichiers absents du Hub", flush=True)
    if args.dry_run:
        return
    for start in range(0, len(uploads), args.chunk):
        chunk = uploads[start : start + args.chunk]
        api.create_commit(
            repo_id=args.repo,
            repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=u.path_in_repo, path_or_fileobj=str(u.local)) for u in chunk],
            commit_message=f"{args.brick}: drain {start + len(chunk)}/{len(uploads)}",
        )
        print(f"  {start + len(chunk)}/{len(uploads)} envoyés", flush=True)
    manifest = args.out / "manifest.drained.jsonl"
    write_manifest(rewritten, manifest)
    extras = [CommitOperationAdd(path_in_repo=f"{args.brick}/manifest.jsonl", path_or_fileobj=str(manifest))]
    for name in ("dropped.jsonl", "summary.json"):
        if (args.out / name).exists():
            extras.append(CommitOperationAdd(path_in_repo=f"{args.brick}/{name}", path_or_fileobj=str(args.out / name)))
    api.create_commit(
        repo_id=args.repo,
        repo_type="dataset",
        operations=extras,
        commit_message=f"{args.brick}: manifest after drain ({len(rewritten)} clips)",
    )
    print(json.dumps({"clips": len(rewritten), "uploaded": len(uploads)}), flush=True)


if __name__ == "__main__":
    main()
