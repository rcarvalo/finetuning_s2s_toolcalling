#!/usr/bin/env python3
"""Voice the v5 refusals with Qwen3-TTS, then ship them to the Hub.

The assistant track of this corpus is Qwen3-TTS speaker **Aiden** (commit
1d5c19a): in interleaved Phase B training the model learns to PREDICT the
assistant answers' audio codes, so that voice becomes the model's own. The
refusals are assistant turns, so they must match it — the user turns are
Kokoro, and that split is deliberate (input rewards diversity, output rewards
fidelity).

v5 rewrites every "the payload does not answer that" turn into a situated
refusal naming what was found and what was asked. Each one is distinct, so each
needs its own clip — v4 could share five WAVs because it had five fixed
sentences.

This path needs NONE of the vLLM stack, which is what makes it usable: the
Voxtral route dies on `StageEngineCoreProc died during READY`, the same failure
that blocks the serving engine.

Resumable: existing WAVs are kept, and a partial tarball goes to the Hub so a
lost machine never costs the accumulated audio twice.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import soundfile as sf
import torch

REPO = "Rcarvalo/tc-en-voice-agent-v1"
TAG = os.environ.get("TC_EN_TAG", "v5")
MANIFEST_IN_REPO = f"phase_b/miss_to_tts_{TAG}.jsonl"
ARCHIVE_IN_REPO = f"phase_b/miss_audio_{TAG}.tar.gz"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
INSTRUCT = "Speak as a friendly, clear voice assistant: warm, natural pace, no whispering."
AUDIO = Path(f"data/audio_miss_{TAG}")


def fetch_manifest() -> list[dict[str, str]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, MANIFEST_IN_REPO, repo_type="dataset")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"manifest: {len(rows)} refusals", flush=True)
    return rows


def restore_checkpoint() -> None:
    """Resume from the last partial tarball rather than re-synthesizing."""
    if any(AUDIO.glob("*.wav")):
        return
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        tarball = hf_hub_download(REPO, ARCHIVE_IN_REPO, repo_type="dataset")
    except (EntryNotFoundError, Exception):
        return
    with tarfile.open(tarball) as archive:
        archive.extractall(AUDIO, filter="data")
    print(f"restored {len(list(AUDIO.glob('*.wav')))} wavs from the Hub", flush=True)


def push_tarball() -> None:
    archive = Path(f"miss_audio_{TAG}.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for wav in sorted(AUDIO.glob("*.wav")):
            tar.add(wav, arcname=wav.name)
    from huggingface_hub import HfApi

    HfApi().upload_file(path_or_fileobj=str(archive), path_in_repo=ARCHIVE_IN_REPO, repo_id=REPO, repo_type="dataset")


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    restore_checkpoint()

    rows = fetch_manifest()
    pending = [row for row in rows if not (AUDIO / row["audio"]).exists()]
    print(f"{len(pending)} to synthesize", flush=True)
    if not pending:
        push_tarball()
        print("MISSES_V5_DONE", flush=True)
        return

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16)
    speaker = os.environ.get("ASSISTANT_VOICE", "Aiden")
    batch_size = int(os.environ.get("TTS_BATCH", "8"))
    print(f"speaker={speaker} batch={batch_size}", flush=True)

    done = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [row["text"] for row in batch]
        wavs, rate = model.generate_custom_voice(
            text=texts,
            language=["English"] * len(texts),
            speaker=[speaker] * len(texts),
            instruct=[INSTRUCT] * len(texts),
        )
        for row, wav in zip(batch, wavs, strict=True):
            sf.write(str(AUDIO / row["audio"]), wav, rate, subtype="PCM_16")
        done += len(batch)
        if done % 48 < batch_size or done == len(pending):
            print(f"  {done}/{len(pending)}", flush=True)

    push_tarball()
    voiced = len(list(AUDIO.glob("*.wav")))
    print(f"packed {voiced}/{len(rows)} refusal clips", flush=True)
    if voiced != len(rows):
        raise SystemExit(f"only {voiced} of {len(rows)} refusals were voiced")
    print("MISSES_V5_DONE", flush=True)


if __name__ == "__main__":
    main()
