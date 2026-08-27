#!/usr/bin/env python3
"""Re-voice the scenario user turns with Kokoro, locally, and ship them to the Hub.

The scenario INPUT audio is neither tracked in git nor on the Hub, so a fresh
pod finds `data/audio_scenarios` empty and the gate run dies on the first turn.
It is user-side audio, so Kokoro is the right engine — and it runs on CPU,
which means no GPU and no cost.

Each turn is rendered with the voice `data/scenarios_voiced.jsonl` recorded for
it, not a fresh draw: v4 was measured on those exact voices, and the whole
point of this run is comparability.

    python infra/voice_scenarios.py --push
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

VOICED = Path("data/scenarios_voiced.jsonl")
AUDIO = Path("data/audio_scenarios")
REPO = "Rcarvalo/tc-en-voice-agent-v1"
ARCHIVE_IN_REPO = "scenarios/audio_scenarios.tar.gz"
SAMPLE_RATE = 16_000


def turns() -> list[tuple[str, str, str]]:
    """(relative wav name, text, Kokoro voice) for every user turn."""
    jobs = []
    for line in VOICED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        dialogue = json.loads(line)
        for turn in dialogue["turns"]:
            if turn.get("role") == "user" and turn.get("audio"):
                jobs.append((turn["audio"], turn["text"], turn.get("voice", "af_heart")))
    return jobs


def render(jobs: list[tuple[str, str, str]]) -> int:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a")
    written = 0
    for name, text, voice in jobs:
        target = AUDIO / name
        if target.exists():
            continue
        chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
        wav24 = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])
        # Kokoro is 24 kHz native; the mel encoder is calibrated at 16 kHz, and
        # a mismatch degrades what the model hears without raising anything.
        wav16 = torchaudio.functional.resample(torch.from_numpy(wav24), 24_000, SAMPLE_RATE).numpy()
        sf.write(str(target), wav16, SAMPLE_RATE, subtype="PCM_16")
        written += 1
        print(f"  {name} ({voice})", flush=True)
    return written


def push() -> None:
    from huggingface_hub import HfApi

    archive = Path("audio_scenarios.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for wav in sorted(AUDIO.glob("*.wav")):
            tar.add(wav, arcname=wav.name)
    HfApi().upload_file(path_or_fileobj=str(archive), path_in_repo=ARCHIVE_IN_REPO, repo_id=REPO, repo_type="dataset")
    print(f"pushed {ARCHIVE_IN_REPO}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="upload the tarball to the Hub")
    args = parser.parse_args()

    AUDIO.mkdir(parents=True, exist_ok=True)
    jobs = turns()
    print(f"{len(jobs)} scenario turns", flush=True)
    written = render(jobs)
    print(f"{written} rendered, {len(list(AUDIO.glob('*.wav')))} present", flush=True)
    if args.push:
        push()


if __name__ == "__main__":
    main()
