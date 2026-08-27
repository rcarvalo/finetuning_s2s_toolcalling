"""Publish one brick of the FR corpus to its folder in the Hub repository.

Entry point: ``lfm2-corpus-push``. One brick at a time, on purpose: bricks are
topped up individually when a gate says which one is short, and re-uploading
the whole corpus to fix one of them wastes hours of bandwidth.

    lfm2-corpus-push --brick A --local data/corpus/A_assistant_speech \\
      --repo-id Rcarvalo/lfm25-fr-corpus-v1

The local folder must hold ``manifest.jsonl`` and ``audio/``; the manifest is
validated before anything is uploaded, so a malformed corpus never reaches the
Hub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

from lfm2_audio.data_prep.corpus_layout import (
    AUDIO_DIR,
    BRICKS_BY_KEY,
    MANIFEST_NAME,
    CorpusError,
    brick_readme,
    read_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brick", required=True, choices=sorted(BRICKS_BY_KEY))
    parser.add_argument("--local", required=True, type=Path, help="dossier local de la brique")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", help="valider sans publier")
    args = parser.parse_args()

    brick = BRICKS_BY_KEY[args.brick]
    manifest = args.local / MANIFEST_NAME
    if not manifest.exists():
        print(f"❌ {manifest} introuvable", file=sys.stderr)
        raise SystemExit(1)

    try:
        entries = list(read_manifest(manifest))
        for entry in entries:
            entry.validate()
            if not (args.local / entry.audio).exists():
                message = f"audio manquant pour {entry.id} : {entry.audio}"
                raise CorpusError(message)
    except CorpusError as error:
        print(f"❌ manifeste invalide — rien n'est publié : {error}", file=sys.stderr)
        raise SystemExit(1) from error

    hours = sum(entry.duration_s for entry in entries) / 3600
    languages = sorted({entry.lang for entry in entries})
    print(f"{brick.folder} : {len(entries)} clips · {hours:.2f} h · langues {languages}")

    (args.local / "README.md").write_text(brick_readme(brick, len(entries), hours), encoding="utf-8")
    if args.dry_run:
        print("(dry-run) validation seule, rien n'a été publié")
        return

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.local),
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=brick.folder,
        allow_patterns=[MANIFEST_NAME, "README.md", f"{AUDIO_DIR}/**"],
        commit_message=f"{brick.folder}: {len(entries)} clips ({hours:.1f} h)",
    )
    print(f"→ hf.co/datasets/{args.repo_id}/tree/main/{brick.folder}")


if __name__ == "__main__":
    main()
