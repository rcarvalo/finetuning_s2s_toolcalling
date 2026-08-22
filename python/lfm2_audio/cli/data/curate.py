"""Merge dialogue sources into one clean corpus (weekend steps 4-5).

Entry point: ``lfm2-dataset-curate``.
Logic lives in :mod:`lfm2_audio.data_prep.curation` — this module is CLI only.

Deduplicates on the normalized user utterance and refuses anything that also
appears in the held-out split, so the step-7 comparison cannot be a
memorization check.

    lfm2-dataset-curate \
        --source data/tc_en_train.jsonl --source data/tc_en_s2s.jsonl \
        --held-out benchmark/toolcalling_en/cases.sample.jsonl \
        --out data/curated/train.jsonl

Exit code is 1 when leakage was found, so a pipeline can stop on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=Path,
        help="JSONL of dialogues; repeat it. Order matters: the first occurrence of an utterance wins.",
    )
    parser.add_argument(
        "--held-out",
        action="append",
        default=[],
        type=Path,
        help="JSONL whose utterances must NOT appear in the output (test/benchmark split).",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--allow-leakage",
        action="store_true",
        help="report leakage instead of failing (default: exit 1 so a pipeline stops)",
    )
    args = parser.parse_args()

    from lfm2_audio.data_prep.curation import curate, load_jsonl, write_jsonl

    held_out = [dialogue for path in args.held_out for dialogue in load_jsonl(path)]
    sources = {str(path): list(load_jsonl(path)) for path in args.source}

    kept, report = curate(sources, held_out=held_out)
    write_jsonl(kept, args.out)
    print(report.summary())
    print(f"written: {args.out}")

    if report.leaked and not args.allow_leakage:
        print(f"ERROR: {report.leaked} dialogue(s) leaked from the held-out split", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
