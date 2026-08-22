"""Compare two evaluation reports — baseline vs fine-tune (weekend step 7).

Entry point: ``lfm2-eval-compare``.
Logic lives in :mod:`lfm2_audio.evaluation.comparison` — this module is CLI only.

    lfm2-eval-compare --baseline reports/baseline_en_audio.json \
        --candidate reports/ft_v1_en_audio.json --out reports/compare_v1.md

Exit code is 1 when a metric regressed, so a training loop can gate on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", default=None, type=Path, help="write the Markdown table here")
    args = parser.parse_args()

    from lfm2_audio.evaluation.comparison import compare_files

    comparison = compare_files(args.baseline, args.candidate)
    markdown = comparison.to_markdown(baseline_name=args.baseline.stem, candidate_name=args.candidate.stem)
    print(markdown)
    if args.out:
        args.out.write_text(markdown, encoding="utf-8")

    return 1 if comparison.regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
