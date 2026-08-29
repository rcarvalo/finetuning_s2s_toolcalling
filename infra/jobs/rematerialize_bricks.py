"""Rebuild brick audio from saved manifests, and push it to the Hub.

The overnight pods died with their disks (balance exhaustion, by design), but
their real product survived: the *manifests* — which clip, from where, with
what verified WER. Audio is re-derivable: brick A-stock clips are files in the
source dataset, brick B clips are rows of a deterministic stream. This job
re-materialises both and streams them to the corpus repo, no GPU needed.

No re-verification: the manifests carry the WERs a GPU already paid for.
Identity is checked cheaply instead — brick B rows must match the manifest's
transcript, or the stream shifted and the row is skipped.

  A_stock        25.75 h, 14 295 clips — same fr_female voice as the synthesis
                 (VERSA spk_similarity 0.769 vs a 0.740 same-voice control)
  B_user_speech  11.8 h, 9 654 clips, 901 speakers, hold-out applied
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out"))

sys.path.insert(0, str(ROOT / "python"))

BRICKS = os.environ.get("REMAT_BRICKS", "A_stock,B_user_speech").split(",")
CORPUS_REPO = os.environ.get("REMAT_REPO", "Rcarvalo/lfm25-fr-corpus-v1")
DIALOGUE_REPO = "Rcarvalo/french-dialogue-tts-1000h"
CV_REPO = "baptistefrancois1/s2s-fr-finetuning"
PUSH_EVERY_FILES = int(os.environ.get("REMAT_PUSH_EVERY", "800"))


def ensure_deps() -> None:
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets>=3.0", "soxr>=0.5"], check=False)


def load_manifest(brick: str):  # noqa: ANN201 — list[dict]
    """The manifest published from the Mac's checkpoint of the dead pod."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CORPUS_REPO, f"{brick}/manifest.jsonl", repo_type="dataset")
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def rematerialize_a_stock(pusher) -> dict:  # noqa: ANN001
    """Stock clips are plain files in the source dataset — chunked snapshots."""
    from huggingface_hub import snapshot_download

    entries = load_manifest("A_stock")
    audio_dir = OUT / "A_stock" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # id a_<stem> → <stem>.wav in the source repo. Chunked allow_patterns: one
    # giant pattern list made the matcher quadratic (24 min of CPU on the pod).
    wanted = {entry["id"]: f"{entry['id'][2:]}.wav" for entry in entries}
    todo = {cid: name for cid, name in wanted.items() if not (audio_dir / f"{cid}.wav").exists()}
    print(f"[A_stock] {len(entries)} clips, {len(todo)} à rematérialiser", flush=True)

    names = list(todo.items())
    done = 0
    for start in range(0, len(names), 500):
        chunk = dict(names[start : start + 500])
        # Wildcard prefix: the manifest keeps only the stem, and the source
        # repo may store its wavs under a subdirectory.
        local = Path(
            snapshot_download(DIALOGUE_REPO, repo_type="dataset", allow_patterns=[f"*{n}" for n in chunk.values()])
        )
        by_name = {p.name: p for p in local.rglob("*.wav")}
        for cid, name in chunk.items():
            source = by_name.get(name)
            if source is not None:
                (audio_dir / f"{cid}.wav").write_bytes(source.read_bytes())
                done += 1
        print(f"[A_stock] {done}/{len(todo)}", flush=True)
        if done % PUSH_EVERY_FILES < 500:
            pusher.push(message=f"A_stock: {done}/{len(todo)} rematérialisés")
    return {"brick": "A_stock", "clips": done}


def rematerialize_b(pusher) -> dict:  # noqa: ANN001
    """Brick B rows come back from the same deterministic stream, by index.

    The id encodes the stream position (``b_%06d``); the transcript in the
    manifest is the identity check — a mismatch means the source dataset
    changed order, and the row is skipped rather than mislabelled.
    """
    import io

    import numpy as np
    import soundfile as sf
    import soxr
    from datasets import Audio, load_dataset

    entries = load_manifest("B_user_speech")
    audio_dir = OUT / "B_user_speech" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    by_index = {int(entry["id"][2:]): entry for entry in entries}
    missing = {i for i, entry in by_index.items() if not (audio_dir / f"{entry['id']}.wav").exists()}
    print(f"[B] {len(entries)} clips, {len(missing)} à rematérialiser, index max {max(by_index)}", flush=True)
    if not missing:
        return {"brick": "B_user_speech", "clips": 0, "mismatches": 0}

    rows = load_dataset(CV_REPO, "common_voice_fr", split="train", streaming=True)
    rows = rows.cast_column("audio", Audio(decode=False))
    done = mismatches = 0
    top = max(missing)
    for index, row in enumerate(rows):
        if index > top:
            break
        if index not in missing:
            continue
        entry = by_index[index]
        text = " ".join(str(row.get("transcript", "") or "").split())
        if text != entry["text"]:
            mismatches += 1
            continue
        data, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if rate != 16_000:
            data = soxr.resample(data, rate, 16_000)
        sf.write(str(audio_dir / f"{entry['id']}.wav"), data, 16_000, subtype="PCM_16")
        done += 1
        if done % PUSH_EVERY_FILES == 0:
            print(f"[B] {done}/{len(missing)}", flush=True)
            pusher.push(message=f"B_user_speech: {done}/{len(missing)} rematérialisés")
    if mismatches:
        print(f"[B] {mismatches} lignes écartées (transcript décalé — le flux source a bougé)", flush=True)
    return {"brick": "B_user_speech", "clips": done, "mismatches": mismatches}


def main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ensure_deps()

    from lfm2_audio.data_prep.streaming_push import StreamingPusher

    results = []
    for brick in [b.strip() for b in BRICKS]:
        target = OUT / brick
        target.mkdir(parents=True, exist_ok=True)
        pusher = StreamingPusher(target, CORPUS_REPO, brick)
        pusher.verify()
        builder = {"A_stock": rematerialize_a_stock, "B_user_speech": rematerialize_b}.get(brick)
        if builder is None:
            continue
        results.append(builder(pusher))
        pusher.push(message=f"{brick}: rematérialisation finale")
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    print("===RESULT rematerialize===", flush=True)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    print("===END===", flush=True)


if __name__ == "__main__":
    main()
