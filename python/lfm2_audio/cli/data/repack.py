"""Repack a spoken tool-calling dataset with an evaluation split worth trusting.

Entry point: ``lfm2-dataset-repack``.

The shipped corpus pairs 2930 training rows with a 12-row test split. Twelve
cases proved the vanilla model never calls a tool, but cannot separate a real
gain from noise once fine-tuning starts. This command carves a larger held-out
split out of the training rows — stratified on the tool, deduplicated on the
utterance — and keeps the original held-out-voice split intact beside it.

The two evaluation splits answer different questions and must stay separate:

* ``test_utterances`` — unseen wording, voices already seen. Statistical power.
* ``test_voices`` — the original 12, voices never seen in training. Generalization.

    lfm2-dataset-repack --source Rcarvalo/tc-en-audio-toolcalling \
        --target Rcarvalo/tc-en-voice-agent-v1 --test-size 200

Audio is copied as stored bytes: nothing is decoded, so no codec dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _target_of_row(row: dict[str, Any]) -> str:
    """Behaviour a flat Hub row exercises — the tool called, or ``"none"``."""
    return str(row.get("tool_name") or "none")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="Rcarvalo/tc-en-audio-toolcalling")
    parser.add_argument("--target", required=True, help="dataset repo to create/update")
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--public", action="store_true", help="publish publicly (default: private)")
    parser.add_argument("--out-dir", default=Path("data/repack"), type=Path)
    parser.add_argument("--dry-run", action="store_true", help="write the parquets locally, do not push")
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    from lfm2_audio.data_prep.curation import normalize_utterance
    from lfm2_audio.data_prep.splitting import stratified_split

    def read(split: str) -> list[dict[str, Any]]:
        path = hf_hub_download(args.source, f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        return pq.read_table(path).to_pylist()

    train_rows, voice_rows = read("train"), read("test")

    # Copies of one utterance must never straddle the split, so drop them first.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in train_rows:
        key = normalize_utterance(str(row.get("utterance") or ""))
        if key and key not in seen:
            seen.add(key)
            unique.append(row)
    duplicates = len(train_rows) - len(unique)

    train, test, report = stratified_split(unique, test_size=args.test_size, seed=args.seed, target=_target_of_row)
    print(f"source {args.source}: {len(train_rows)} rows, {duplicates} duplicate utterance(s) dropped")
    print(report.summary())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, rows in (("train", train), ("test_utterances", test), ("test_voices", voice_rows)):
        path = args.out_dir / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        written[name] = path
        print(f"  {name}: {len(rows)} rows → {path}")

    if args.dry_run:
        print("dry run: nothing pushed")
        return 0

    api = HfApi()
    api.create_repo(args.target, repo_type="dataset", private=not args.public, exist_ok=True)
    for name, path in written.items():
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"data/{name}.parquet",
            repo_id=args.target,
            repo_type="dataset",
        )
    print(f"pushed to https://huggingface.co/datasets/{args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
