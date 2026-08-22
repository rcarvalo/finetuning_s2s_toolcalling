"""Prépare les données d'entraînement sur le pod (idempotent, relançable).

Miroir de ``notebooks/finetune_toolcalling.ipynb`` : réhydrate le dataset HF
audio → split train/val déterministe → packing ``preprocess_sft``. Chaque étape
est sautée si sa sortie existe déjà : ``sky exec`` peut relancer un entraînement
sans repayer la préparation.

    python infra/prepare_data.py --repo-id Rcarvalo/tc-en-audio-toolcalling
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TRAIN_DATASET = Path("datasets/tc_en_train")
VAL_DATASET = Path("datasets/tc_en_val")
TOOLS_FILE = Path("tools_en.json")


def rehydrate_if_missing(repo_id: str, out: Path) -> None:
    if (out / "train.jsonl").exists():
        print(f"réhydratation sautée : {out}/train.jsonl existe")
        return
    from lfm2_audio.cli.hf_to_dialogues import rehydrate

    counts = rehydrate(repo_id, out)
    print(f"réhydraté {repo_id} → {out} : {counts}")


def split_if_missing(out: Path, val_frac: float, seed: int) -> None:
    train_only, val = out / "train_only.jsonl", out / "val.jsonl"
    if train_only.exists() and val.exists():
        print("split sauté : train_only.jsonl + val.jsonl existent")
        return
    rows = (out / "train.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    random.Random(seed).shuffle(rows)
    k = max(1, int(len(rows) * val_frac))
    val.write_text("".join(rows[:k]), encoding="utf-8")
    train_only.write_text("".join(rows[k:]), encoding="utf-8")
    print(f"split : train {len(rows) - k} / val {k}")


def write_tool_definitions() -> None:
    from lfm2_audio.tools.schemas import TOOLCALLING_EN_TOOL_DEFINITIONS

    TOOLS_FILE.write_text(json.dumps(TOOLCALLING_EN_TOOL_DEFINITIONS), encoding="utf-8")


def preprocess_if_missing(out: Path) -> None:
    from lfm2_audio.cli.preprocess_sft import main as preprocess

    for split, dataset_dir in (("train_only", TRAIN_DATASET), ("val", VAL_DATASET)):
        if dataset_dir.exists():  # preprocess_sft exige un dossier neuf → on ne re-packe jamais par-dessus
            print(f"packing sauté : {dataset_dir} existe")
            continue
        preprocess(
            [
                "--dialogues",
                str(out / f"{split}.jsonl"),
                "--audio-root",
                str(out / "audio_train"),
                "--output",
                str(dataset_dir),
                "--tool-definitions",
                str(TOOLS_FILE),
                "--assistant-audio-mode",
                "sequential",
            ]
        )
        print(f"packé {split} → {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="dataset HF audio (push par build_hf_dataset)")
    parser.add_argument("--out", default="data/tc_en")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    rehydrate_if_missing(args.repo_id, out)
    split_if_missing(out, args.val_frac, args.seed)
    write_tool_definitions()
    preprocess_if_missing(out)


if __name__ == "__main__":
    main()
