"""Inventory the datasets of a Hub author (weekend step 3).

Entry point: ``lfm2-dataset-inventory``.
Logic lives in :mod:`lfm2_audio.data_prep.hub_inventory` — this module is CLI only.

    lfm2-dataset-inventory --author Rcarvalo --out docs/dataset_inventory.md
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--author", default="Rcarvalo")
    parser.add_argument("--out", default=Path("docs/dataset_inventory.md"), type=Path)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    from lfm2_audio.data_prep.hub_inventory import write_inventory

    entries = write_inventory(HfApi(), args.author, args.out)
    private = sum(e.private for e in entries)
    print(f"{len(entries)} dataset(s) ({private} private) → {args.out}")


if __name__ == "__main__":
    main()
