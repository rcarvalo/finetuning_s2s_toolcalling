#!/usr/bin/env python3
"""Voice the v5 refusals with Voxtral, then ship them to the Hub.

Runs inside the TTS pod, after ``pod_tts_job.sh`` has a Voxtral server up.

v5 rewrites every "the payload does not answer that" turn into a situated
refusal naming what was found and what was asked. Each one is distinct, so each
needs its own clip — v4 could share five WAVs because it had five fixed
sentences.

The voice MUST be ``neutral_female``, the assistant voice ``pod_synth_phase_b``
used for the whole corpus. v4 voiced its refusals with Aiden instead, so the
model heard two different voices for itself.

Input is the manifest ``prepare_v5.py --stage transform`` wrote; it is fetched
from the Hub so the pod needs nothing but its own repo clone.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = "Rcarvalo/tc-en-voice-agent-v1"
MANIFEST_IN_REPO = "phase_b/miss_to_tts_v5.jsonl"
ARCHIVE_IN_REPO = "phase_b/miss_audio_v5.tar.gz"
VOICE = "neutral_female"

WORK = Path("/work_misses_v5")
RAW = WORK / "raw"
FINAL = WORK / "final"


def fetch_manifest() -> list[dict[str, str]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, MANIFEST_IN_REPO, repo_type="dataset")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"manifest: {len(rows)} refusals to voice", flush=True)
    return rows


def write_source(rows: list[dict[str, str]]) -> Path:
    """One assistant turn per dialogue — a user turn would be voiced too."""
    src = WORK / "src.jsonl"
    src.write_text(
        "".join(
            json.dumps(
                {
                    "id": row["id"],
                    "turns": [{"role": "assistant", "text": row["text"]}],
                    "meta": {"target_audio": row["audio"]},
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return src


def synthesize(src: Path) -> Path:
    out = WORK / "voiced.jsonl"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lfm2_audio.cli.data.synthesize",
            "--engine",
            "voxtral",
            "--dialogues",
            str(src),
            "--audio-root",
            str(RAW),
            "--out",
            str(out),
            "--voices",
            VOICE,
            "--assistant-voice",
            VOICE,
            "--concurrency",
            "8",
        ],
        check=True,
    )
    return out


def rename_and_pack(voiced: Path) -> int:
    """Rename to the names prepare_v5 wrote into the dialogues.

    A clip under any other name leaves its refusal silent at packing time —
    after the synthesis has been paid for.
    """
    kept = 0
    for line in voiced.read_text(encoding="utf-8").splitlines():
        dialogue = json.loads(line)
        target = (dialogue.get("meta") or {}).get("target_audio")
        source = next((t.get("audio") for t in dialogue["turns"] if t["role"] == "assistant" and t.get("audio")), None)
        if not (target and source and (RAW / source).exists()):
            print(f"MISSING {dialogue['id']} source={source}", flush=True)
            continue
        shutil.copy(RAW / source, FINAL / target)
        kept += 1

    archive = WORK / "miss_audio_v5.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for wav in sorted(FINAL.glob("*.wav")):
            tar.add(wav, arcname=wav.name)

    from huggingface_hub import HfApi

    HfApi().upload_file(path_or_fileobj=str(archive), path_in_repo=ARCHIVE_IN_REPO, repo_id=REPO, repo_type="dataset")
    return kept


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)

    rows = fetch_manifest()
    kept = rename_and_pack(synthesize(write_source(rows)))
    print(f"packed {kept}/{len(rows)} refusal clips", flush=True)
    if kept != len(rows):
        raise SystemExit(f"only {kept} of {len(rows)} refusals were voiced")
    print("MISSES_V5_DONE", flush=True)


if __name__ == "__main__":
    main()
