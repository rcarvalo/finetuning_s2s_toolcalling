"""Build the v5 training set: v4's recipe with realistic payloads.

Runs on the training box. Idempotent. Two stages, because the middle one needs
a GPU:

  --stage transform   rewrite the Phase B payloads, emit the TTS manifest
  (then: infra/sky_tts.yaml voices data/tc_en_v5/miss_to_tts.jsonl)
  --stage pack        merge with Phase A, split, pack for training

Starts from ``phase_b/train.jsonl`` — the ORIGINAL payloads — not from
``train_v4.jsonl``, which already carries v4's noise. Stacking both would
bury the answer twice over and measure nothing.

What changes against v4, and why (see docs/v4_report.md and the live demo):
  * distractors are drawn on topic, not at random — v4 let topic alone isolate
    the answering entry, a shortcut that does not transfer;
  * web results are prose, not fields — DuckDuckGo returns 400 characters with
    the fact buried, and v4 refused on a payload that answered four times over;
  * refusals name what was found and what was asked — v4's five templates could
    be written without reading the payload, which taught refusal-by-reflex;
  * miss_ratio 0.15 -> 0.08, against that same over-refusal.

The refusal clips must be voiced ``neutral_female``, the assistant voice
``pod_synth_phase_b.sh`` used for the whole corpus. v4 voiced them with Aiden
instead, so the model heard two different voices for itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

REPO = "Rcarvalo/tc-en-voice-agent-v1"
TAG = os.environ.get("TC_EN_TAG", "v5")
"""Which corpus this builds: v5, v5_1… Every Hub path and local dir derives from it."""
OUT = Path(f"data/tc_en_{TAG}")
AUDIO = OUT / "audio"
VAL_SIZE = 150
ASSISTANT_VOICE = "neutral_female"


def fetch_phase_a() -> list[dict[str, Any]]:
    """Phase A stays exactly as v4 had it: single-turn, already voiced."""
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    from lfm2_audio.data_prep.hf_rehydrate import row_to_dialogue

    shards = sorted(
        name
        for name in HfApi().list_repo_files(REPO, repo_type="dataset")
        if name.startswith("data/train") and name.endswith(".parquet")
    )
    if not shards:
        raise FileNotFoundError(f"no Phase A parquet under data/ in {REPO}")

    rows: list[dict[str, Any]] = []
    for shard in shards:
        rows.extend(pq.read_table(hf_hub_download(REPO, shard, repo_type="dataset")).to_pylist())
    dialogues = []
    for row in rows:
        rel = row["audio"]["path"]
        if not (AUDIO / rel).exists():
            data, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            sf.write(str(AUDIO / rel), data, rate, subtype="PCM_16")
        dialogues.append(row_to_dialogue(row, rel))
    print(f"phase A: {len(dialogues)} single-turn dialogues", flush=True)
    return dialogues


def fetch_phase_b_original() -> list[dict[str, Any]]:
    """Phase B with its ORIGINAL payloads, plus the audio that already exists."""
    from huggingface_hub import hf_hub_download

    jsonl = hf_hub_download(REPO, "phase_b/train.jsonl", repo_type="dataset")
    tarballs = [
        hf_hub_download(REPO, "phase_b/audio.tar.gz", repo_type="dataset"),
        hf_hub_download(REPO, "phase_b/assistant_audio.tar.gz", repo_type="dataset"),
    ]

    marker = AUDIO / ".phase_b_extracted"
    if not marker.exists():
        for tarball in tarballs:
            with tarfile.open(tarball) as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    # Both corpora number ids ``tc_NNNNNN_*``: without the
                    # prefix the filenames collide and cross the audio silently.
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
    print(f"phase B: {len(dialogues)} conversational dialogues (original payloads)", flush=True)
    return dialogues


def transform(dialogues: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Apply the v5 payload rewrite; return the dialogues and what needs voicing.

    Every refusal is now distinct, so each needs its OWN clip — v4 could share
    five WAVs because it had five fixed sentences.
    """
    from lfm2_audio.data_prep.payload_realism import PayloadRealism

    rewritten, misses = PayloadRealism().apply(dialogues)
    manifest: list[dict[str, str]] = []
    for dialogue in rewritten:
        if not (dialogue.get("meta") or {}).get("answer_absent"):
            continue
        for turn in dialogue["turns"]:
            if turn.get("role") == "assistant" and turn.get("text") and not turn.get("tool_calls"):
                name = f"pb_miss_{TAG}_{dialogue['id']}.wav"
                turn["audio"] = name
                manifest.append({"id": dialogue["id"], "text": turn["text"], "audio": name})
    print(f"payloads rewritten: {len(rewritten)} dialogues, {misses} refusals to voice", flush=True)
    return rewritten, manifest


def stage_transform() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    phase_b, manifest = transform(fetch_phase_b_original())

    (OUT / f"phase_b_{TAG}.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in phase_b), encoding="utf-8"
    )
    (OUT / "miss_to_tts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8"
    )
    print(f"\nwrote {OUT}/phase_b_{TAG}.jsonl and {OUT}/miss_to_tts.jsonl", flush=True)
    print(f"next: voice {len(manifest)} refusals as '{ASSISTANT_VOICE}', into {AUDIO}", flush=True)
    # Shipped to the Hub right here: the pack stage runs on another machine
    # (it needs liquid-audio) and must find these without any synced workdir.
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(OUT / f"phase_b_{TAG}.jsonl"),
        path_in_repo=f"phase_b/train_{TAG}.jsonl",
        repo_id=REPO,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(OUT / "miss_to_tts.jsonl"),
        path_in_repo=f"phase_b/miss_to_tts_{TAG}.jsonl",
        repo_id=REPO,
        repo_type="dataset",
    )
    print(f"pushed phase_b/train_{TAG}.jsonl and phase_b/miss_to_tts_{TAG}.jsonl", flush=True)
    print("TRANSFORM_DONE", flush=True)


def fetch_v5_artifacts() -> None:
    """Pull what --stage transform produced, plus the voiced refusals.

    Packing needs liquid-audio, which the laptop does not have: the transform
    runs there, the pack runs on the training machine. So the pod must be able
    to rebuild the whole corpus from the Hub alone, exactly as v4 did.
    """
    from huggingface_hub import hf_hub_download

    target = OUT / f"phase_b_{TAG}.jsonl"
    if not target.exists():
        source = hf_hub_download(REPO, f"phase_b/train_{TAG}.jsonl", repo_type="dataset")
        target.write_bytes(Path(source).read_bytes())
        print(f"fetched phase_b/train_{TAG}.jsonl -> {target}", flush=True)

    if not any(AUDIO.glob(f"pb_miss_{TAG}_*.wav")):
        tarball = hf_hub_download(REPO, f"phase_b/miss_audio_{TAG}.tar.gz", repo_type="dataset")
        with tarfile.open(tarball) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                assert extracted is not None
                (AUDIO / Path(member.name).name).write_bytes(extracted.read())
        print(f"fetched {len(list(AUDIO.glob(f'pb_miss_{TAG}_*.wav')))} refusal clips", flush=True)

    # The Phase B audio that already existed: user turns and unchanged answers.
    #
    # No marker file guards this. An earlier version used one and the pod
    # skipped the whole extraction: the marker had been written on the laptop,
    # then synced over with the workdir, so it claimed audio the pod did not
    # have. Extraction is idempotent — existing files are left alone — so
    # asking the filesystem every time is both cheaper and honest.
    written = 0
    for name in ("phase_b/audio.tar.gz", "phase_b/assistant_audio.tar.gz"):
        with tarfile.open(hf_hub_download(REPO, name, repo_type="dataset")) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                out = AUDIO / f"pb_{Path(member.name).name}"
                if not out.exists():
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    out.write_bytes(extracted.read())
                    written += 1
    print(f"phase B audio: {written} extracted, {len(list(AUDIO.glob('pb_*.wav')))} present", flush=True)


def stage_pack() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    fetch_v5_artifacts()

    phase_b_path = OUT / f"phase_b_{TAG}.jsonl"
    with phase_b_path.open(encoding="utf-8") as handle:
        phase_b = [json.loads(line) for line in handle]

    # Phase A is rebuilt unconditionally, because fetching it is also what
    # WRITES its audio. Guarding on train_all.jsonl skipped that on the pod —
    # the merged file had been synced from the laptop, so packing died on a
    # Phase A clip that had never been written. Never let a derived artifact
    # decide whether its own prerequisite gets produced.
    merged_path = OUT / "train_all.jsonl"
    merged = fetch_phase_a() + phase_b
    merged_path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in merged), encoding="utf-8")
    print("merged corpus ready", flush=True)

    # Every referenced clip, not just the refusals: fail here rather than an
    # hour later inside the packer, where the cause is buried under a
    # DatasetGenerationError.
    missing = sorted(
        {
            turn["audio"]
            for dialogue in merged
            for turn in dialogue["turns"]
            if turn.get("audio") and not (AUDIO / turn["audio"]).exists()
        }
    )
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio clips missing, e.g. {missing[:3]}")
    print(f"audio complete: every clip referenced by {len(merged)} dialogues is present", flush=True)

    train_path, val_path = OUT / "train.jsonl", OUT / "val.jsonl"
    if not (train_path.exists() and val_path.exists()):
        import shutil

        from lfm2_audio.data_prep.splitting import stratified_split

        with merged_path.open(encoding="utf-8") as handle:
            dialogues = [json.loads(line) for line in handle]
        train, val, report = stratified_split(dialogues, test_size=VAL_SIZE, seed=3)
        print(report.summary(), flush=True)
        for path, rows in ((train_path, train), (val_path, val)):
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        # The in-training scorer reads audio from its own root.
        val_audio = OUT / "audio_val"
        val_audio.mkdir(exist_ok=True)
        for row in val:
            rel = next((t.get("audio") for t in row["turns"] if t.get("audio")), None)
            if rel and (AUDIO / rel).exists():
                shutil.copy(AUDIO / rel, val_audio / rel)

    from lfm2_audio.tools import schemas

    Path("tools_en.json").write_text(json.dumps(schemas.TOOLCALLING_EN_TOOL_DEFINITIONS), encoding="utf-8")

    for split, dataset_dir in (("train", f"datasets/tc_en_{TAG}_train"), ("val", f"datasets/tc_en_{TAG}_val")):
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
    print("PREPARE_V5_DONE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["transform", "pack"], required=True)
    args = parser.parse_args()
    stage_transform() if args.stage == "transform" else stage_pack()


if __name__ == "__main__":
    main()
