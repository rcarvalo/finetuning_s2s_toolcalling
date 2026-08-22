"""Prepare the curated corpus for training (weekend step 6 prerequisites).

Runs on the GPU box (needs liquid-audio for packing). Idempotent: each stage is
skipped when its output exists, so a reclaimed Colab session costs minutes.

    python infra/prepare_v1.py --repo Rcarvalo/tc-en-voice-agent-v1

Stages
  1. rehydrate the three splits into JSONL + WAV
  2. carve a validation split out of train (the final test is never watched)
  3. pack train and val for the trainer
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

VAL_SIZE = 200


def rehydrate(repo: str, out: Path) -> dict[str, list[dict[str, Any]]]:
    """Parquet → dialogue JSONL + WAV, one directory per split."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    from lfm2_audio.data_prep.hf_rehydrate import row_to_dialogue

    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "test_utterances", "test_voices"):
        jsonl = out / f"{split}.jsonl"
        audio_root = out / f"audio_{split}"
        if jsonl.exists():
            print(f"  {split}: already rehydrated")
            splits[split] = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
            continue

        audio_root.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(hf_hub_download(repo, f"data/{split}.parquet", repo_type="dataset"))
        dialogues = []
        with jsonl.open("w", encoding="utf-8") as handle:
            for row in table.to_pylist():
                rel = f"{row['id']}_u0.wav"
                audio = row["audio"]
                # Parquet stores the encoded bytes; write them through soundfile
                # so the sample rate on disk matches what the encoder expects.
                data, rate = sf.read(_as_buffer(audio), dtype="float32")
                sf.write(str(audio_root / rel), data, rate, subtype="PCM_16")
                dialogue = row_to_dialogue(row, rel)
                dialogues.append(dialogue)
                handle.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
        splits[split] = dialogues
        print(f"  {split}: {len(dialogues)} dialogues → {jsonl}")
    return splits


def _as_buffer(audio: dict[str, Any]) -> io.BytesIO | str:
    """Hub audio column → something soundfile can read."""

    if isinstance(audio, dict) and audio.get("bytes"):
        return io.BytesIO(audio["bytes"])
    if isinstance(audio, dict) and audio.get("path"):
        return audio["path"]
    message = f"unsupported audio cell: {type(audio).__name__}"
    raise TypeError(message)


def carve_validation(out: Path, dialogues: list[dict[str, Any]]) -> None:
    """Split train into train_only + val, stratified on the tool called.

    The final test splits stay untouched: watching them during training would
    turn the step-7 comparison into a self-fulfilling measurement.
    """
    train_path, val_path = out / "train_only.jsonl", out / "val.jsonl"
    if train_path.exists() and val_path.exists():
        print("  validation split: already carved")
        return

    from lfm2_audio.data_prep.splitting import stratified_split

    train, val, report = stratified_split(dialogues, test_size=VAL_SIZE, seed=1)
    print(f"  {report.summary()}")
    for path, rows in ((train_path, train), (val_path, val)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # The scoring callback reads its questions with the same audio root as train.
    (out / "audio_val").mkdir(exist_ok=True)
    import shutil

    for row in val:
        rel = next((t.get("audio") for t in row["turns"] if t.get("audio")), None)
        if rel:
            source = out / "audio_train" / rel
            if source.exists():
                shutil.copy(source, out / "audio_val" / rel)


def pack(out: Path, tools_file: Path) -> None:
    """Pack train and val into the tensor layout the trainer consumes."""
    from lfm2_audio.tools.schemas import TOOLCALLING_EN_TOOL_DEFINITIONS

    tools_file.write_text(json.dumps(TOOLCALLING_EN_TOOL_DEFINITIONS), encoding="utf-8")

    for split, dataset_dir in (("train_only", "datasets/tc_en_v1_train"), ("val", "datasets/tc_en_v1_val")):
        if Path(dataset_dir).exists():
            print(f"  {dataset_dir}: already packed")
            continue
        subprocess.run(
            [
                sys.executable,
                "-m",
                "lfm2_audio.cli.data.pack_sft",
                "--dialogues",
                str(out / f"{split}.jsonl"),
                "--audio-root",
                str(out / "audio_train"),
                "--output",
                dataset_dir,
                "--tool-definitions",
                str(tools_file),
                "--assistant-audio-mode",
                "sequential",
            ],
            check=True,
        )
        print(f"  packed {split} → {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Rcarvalo/tc-en-voice-agent-v1")
    parser.add_argument("--out", default=Path("data/tc_en_v1"), type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print("1. rehydrate")
    splits = rehydrate(args.repo, args.out)
    print("2. validation split")
    carve_validation(args.out, splits["train"])
    print("3. pack")
    pack(args.out, Path("tools_en.json"))
    print("PREPARE_DONE")


if __name__ == "__main__":
    main()
