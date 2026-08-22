#!/usr/bin/env python3
"""Assemble les dialogues TTS en dataset HF (Audio + colonnes) et push sur le Hub.

Entrée : les JSONL de dialogues APRÈS TTS (train + test), au ``dialogue_schema``
single-turn (tour user avec ``audio``, tour assistant tool_calls/text). Sortie :
un ``DatasetDict`` {train, test} avec la feature ``Audio`` 16 kHz, poussé sur le
Hub (privé par défaut). La carte du dataset reste neutre (synthèse + TTS), sans
nommer le moteur de synthèse.

    lfm2-build-dataset --repo-id Rcarvalo/tc-en-audio \
        --train data/tc_en_train.audio.jsonl --test data/tc_en_bench.audio.jsonl \
        --audio-root data/audio_tc_en --private

NB licence : l'audio est synthétisé par un modèle TTS sous CC-BY-NC-4.0 → le
dataset est marqué non-commercial (recherche). Garde le repo privé si tu n'es
pas sûr de tes droits de redistribution.
"""

from __future__ import annotations

from typing import Any

import argparse
import json
import sys
from pathlib import Path

from datasets import Audio, Dataset

from lfm2_audio.data_prep.hf_dataset import (
    DEFAULT_LICENSE,
    dataset_card,
    dialogue_to_row,
    load_rows,
)
from datasets import DatasetDict
from huggingface_hub import HfApi


def build_split(jsonl: str | Path, audio_root: str | Path) -> Any:  # noqa: ANN401 — Dataset HF non typé

    ds = Dataset.from_list(load_rows(jsonl, audio_root))
    return ds.cast_column("audio", Audio(sampling_rate=16_000))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="ex. Rcarvalo/tc-en-audio")
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test", type=Path, default=None)
    ap.add_argument("--audio-root", required=True, type=Path)
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--license", default=DEFAULT_LICENSE)
    args = ap.parse_args()


    splits = {"train": build_split(args.train, args.audio_root)}
    if args.test:
        splits["test"] = build_split(args.test, args.audio_root)
    dd = DatasetDict(splits)
    print({k: v.num_rows for k, v in dd.items()})

    dd.push_to_hub(args.repo_id, private=args.private)

    card = dataset_card(
        args.repo_id, args.license, dd["train"].num_rows, dd.get("test").num_rows if "test" in dd else 0
    )
    HfApi().upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"poussé sur https://huggingface.co/datasets/{args.repo_id} (private={args.private})")


if __name__ == "__main__":
    main()
