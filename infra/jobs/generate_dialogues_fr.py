#!/usr/bin/env python3
"""Generate the dialogue families the corpus audit found missing, shard by shard, to the Hub.

627 of the 695 dialogues in `C_dialogues` have exactly two user turns — there is
almost no history for the model to learn. `lfm2-generate-fr` already knows how
to write what is missing (`--n-deep`, `--n-social`, `--n-en`, `--n-switch`);
it had simply never been asked. Text only, Gemini, CPU runtime.

    LFM2_JOB=generate_dialogues_fr LFM2_ARGS="--deep 3000 --social 3000 --en 1200 --switch 800"

Why shards: the generator flushes every batch to its output file but cannot
resume from it — a lost Colab VM would start the family over, paying every
Gemini call again. Each shard here is a separate generator run whose file goes
to the Hub the moment it exists (`C_dialogues/v2_parts/`), and a shard already
on the Hub is downloaded rather than regenerated. Ids are re-stamped with the
shard index at merge time, because every generator run numbers from zero.

The merged file lands at `C_dialogues/dialogues_v2.jsonl` — the path the
assistant-voice notebook already lists as a future source, so voicing it is
one line in `BRICK_A_SOURCES`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("LFM2_ROOT", "/content/finetuning_s2s_toolcalling"))
CORPUS_REPO = os.environ.get("LFM2_CORPUS_REPO", "Rcarvalo/lfm25-fr-corpus-v1")
PARTS_DIR = Path("C_dialogues/v2_parts")
MERGED = Path("C_dialogues/dialogues_v2.jsonl")
FAMILIES = {"deep": "--n-deep", "social": "--n-social", "en": "--n-en", "switch": "--n-switch"}
"""Family -> the generator flag that requests it; every other family is forced to 0."""


def run(cmd: list[str]) -> None:
    print("===", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def reid(rows: list[dict[str, Any]], family: str, shard: int) -> list[dict[str, Any]]:
    """Stamp the shard into every id: each generator run numbers from zero.

    `c_deep_0012` from shard 3 becomes `c_deep_s03_0012`; two shards can then
    never collide, and the original ordinal stays readable.
    """
    stamped = []
    for row in rows:
        original = str(row.get("id", ""))
        prefix, _, ordinal = original.rpartition("_")
        stamped_row = dict(row)
        stamped_row["id"] = f"{prefix or family}_s{shard:02d}_{ordinal or '0000'}"
        stamped.append(stamped_row)
    return stamped


def depth_distribution(rows: list[dict[str, Any]]) -> dict[int, int]:
    """User turns per dialogue — the number this whole job exists to move."""
    counter: collections.Counter[int] = collections.Counter()
    for row in rows:
        counter[sum(1 for t in row.get("turns", []) if t.get("role") == "user")] += 1
    return dict(sorted(counter.items()))


def hub_files() -> set[str]:
    from huggingface_hub import HfApi

    return set(HfApi().list_repo_files(CORPUS_REPO, repo_type="dataset"))


def push(relative: Path) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(ROOT / "corpus" / relative),
        path_in_repo=str(relative),
        repo_id=CORPUS_REPO,
        repo_type="dataset",
    )
    print(f"pushed {relative} -> {CORPUS_REPO}", flush=True)


def fetch(relative: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local = ROOT / "corpus" / relative
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(Path(hf_hub_download(CORPUS_REPO, str(relative), repo_type="dataset")).read_bytes())
    return local


def generate_shard(family: str, shard: int, size: int, per_call: int) -> Path:
    relative = PARTS_DIR / f"{family}_{shard:02d}.jsonl"
    local = ROOT / "corpus" / relative
    flags = [flag for flag in FAMILIES.values()]
    cmd = [
        sys.executable,
        "-m",
        "lfm2_audio.cli.data.generate_fr",
        "--out",
        str(local),
        "--n-fr",
        "0",
        "--per-call",
        str(per_call),
    ]
    for flag in flags:
        cmd += [flag, str(size if flag == FAMILIES[family] else 0)]
    run(cmd)
    if not local.exists() or not read_rows(local):
        raise SystemExit(f"shard {relative} came back empty — Gemini quota? rerun later, nothing is lost")
    push(relative)
    return local


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", type=int, default=3000, help="long dialogues with anaphoric callbacks")
    parser.add_argument("--social", type=int, default=3000, help="short social exchanges (v3's weakest register)")
    parser.add_argument("--en", type=int, default=1200, help="conversational English, the preservation share")
    parser.add_argument("--switch", type=int, default=800, help="code-switch dialogues on top of the 196 that exist")
    parser.add_argument("--shard-size", type=int, default=300)
    parser.add_argument("--per-call", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "infra" / "jobs"))
    from _gemini_preflight import preflight

    preflight()

    targets = {"deep": args.deep, "social": args.social, "en": args.en, "switch": args.switch}
    existing = hub_files()
    merged: list[dict[str, Any]] = []
    for family, target in targets.items():
        for shard in range((target + args.shard_size - 1) // args.shard_size):
            size = min(args.shard_size, target - shard * args.shard_size)
            relative = PARTS_DIR / f"{family}_{shard:02d}.jsonl"
            if str(relative) in existing:
                local = fetch(relative)
                print(f"shard {relative}: already on the Hub, reused", flush=True)
            else:
                local = generate_shard(family, shard, size, args.per_call)
            merged += reid(read_rows(local), family, shard)
            print(f"===RESULT=== shard family={family} n={shard} rows={len(read_rows(local))}", flush=True)

    write_rows(ROOT / "corpus" / MERGED, merged)
    push(MERGED)
    kinds = collections.Counter((row.get("meta") or {}).get("kind", "?") for row in merged)
    depth = depth_distribution(merged)
    print(
        f"===RESULT=== merged={MERGED} dialogues={len(merged)} by_kind={dict(kinds)} user_turns={depth}",
        flush=True,
    )


if __name__ == "__main__":
    main()
