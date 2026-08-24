"""Build the v3 training set: Phase A (single-turn) + Phase B (conversational).

Runs on the training box (needs liquid-audio for packing). Idempotent.

Sources, both on the Hub under Rcarvalo/tc-en-voice-agent-v1:
  data/train.parquet   — 2729 single-turn rows, Voxtral voices (Phase A)
  phase_b/train.jsonl + phase_b/audio.tar.gz — 2679 conversational dialogues,
  Kokoro voices, assistant answers voiced (Phase B)

Phase B WAVs get a ``pb_`` prefix on merge: both corpora number their ids
``tc_NNNNNN_*`` and filename collisions would silently cross the audio.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

OUT = Path("data/tc_en_v3")
AUDIO = OUT / "audio"
VAL_SIZE = 150


def fetch_phase_a() -> list[dict[str, Any]]:
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    from lfm2_audio.data_prep.hf_rehydrate import row_to_dialogue

    table = pq.read_table(hf_hub_download("Rcarvalo/tc-en-voice-agent-v1", "data/train.parquet", repo_type="dataset"))
    dialogues = []
    for row in table.to_pylist():
        rel = row["audio"]["path"]
        if not (AUDIO / rel).exists():
            data, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            sf.write(str(AUDIO / rel), data, rate, subtype="PCM_16")
        dialogues.append(row_to_dialogue(row, rel))
    print(f"phase A: {len(dialogues)} single-turn dialogues", flush=True)
    return dialogues


def fetch_phase_b() -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    jsonl = hf_hub_download("Rcarvalo/tc-en-voice-agent-v1", "phase_b/train.jsonl", repo_type="dataset")
    tarball = hf_hub_download("Rcarvalo/tc-en-voice-agent-v1", "phase_b/audio.tar.gz", repo_type="dataset")

    marker = AUDIO / ".phase_b_extracted"
    if not marker.exists():
        with tarfile.open(tarball) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                target = AUDIO / f"pb_{Path(member.name).name}"
                if not target.exists():
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    target.write_bytes(extracted.read())
        marker.touch()

    dialogues = []
    with Path(jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            dialogue = json.loads(line)
            for turn in dialogue.get("turns", []):
                if turn.get("audio"):
                    turn["audio"] = f"pb_{turn['audio']}"
            dialogues.append(dialogue)
    print(f"phase B: {len(dialogues)} conversational dialogues", flush=True)
    return dialogues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    AUDIO.mkdir(parents=True, exist_ok=True)
    merged_path = OUT / "train_all.jsonl"
    if not merged_path.exists():
        merged = fetch_phase_a() + fetch_phase_b()
        with merged_path.open("w", encoding="utf-8") as handle:
            for dialogue in merged:
                handle.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
    print("merged corpus ready", flush=True)

    train_path, val_path = OUT / "train.jsonl", OUT / "val.jsonl"
    if not (train_path.exists() and val_path.exists()):
        from lfm2_audio.data_prep.splitting import stratified_split

        with merged_path.open(encoding="utf-8") as handle:
            dialogues = [json.loads(line) for line in handle]
        train, val, report = stratified_split(dialogues, test_size=VAL_SIZE, seed=3)
        print(report.summary(), flush=True)
        for path, rows in ((train_path, train), (val_path, val)):
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        # The in-training scorer reads audio from its own root.
        val_audio = OUT / "audio_val"
        val_audio.mkdir(exist_ok=True)
        import shutil

        for row in val:
            rel = next((t.get("audio") for t in row["turns"] if t.get("audio")), None)
            if rel and (AUDIO / rel).exists():
                shutil.copy(AUDIO / rel, val_audio / rel)

    from lfm2_audio.tools import schemas

    Path("tools_en.json").write_text(json.dumps(schemas.TOOLCALLING_EN_TOOL_DEFINITIONS), encoding="utf-8")

    for split, dataset_dir in (("train", "datasets/tc_en_v3_train"), ("val", "datasets/tc_en_v3_val")):
        if Path(dataset_dir).exists():
            print(f"{dataset_dir}: already packed", flush=True)
            continue
        # Interleaved 6:12 is the Phase B contract (configs/training/phase_b_s2s.yaml):
        # tool-call turns stay text-only, spoken answers interleave text and audio.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "lfm2_audio.cli.data.pack_sft",
                "--dialogues",
                str(OUT / f"{split}.jsonl"),
                "--audio-root",
                str(AUDIO),
                "--output",
                dataset_dir,
                "--tool-definitions",
                "tools_en.json",
                "--assistant-audio-mode",
                "interleaved",
                "--interleaved-text-tokens",
                "6",
                "--interleaved-audio-tokens",
                "12",
            ],
            check=True,
        )
        print(f"packed {split} → {dataset_dir}", flush=True)
    print("PREPARE_V3_DONE", flush=True)


if __name__ == "__main__":
    main()
