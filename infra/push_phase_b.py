"""Push the locally-synthesized Phase B corpus to the Hub (jsonl + WAV tarball).

Run from the repo root on the machine that ran the synthesis:

    python infra/push_phase_b.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

JSONL = Path("data/phase_b_train.jsonl")
AUDIO = Path("data/audio_phase_b")
TARBALL = Path("/tmp/phase_b_audio.tar.gz")
REPO = "Rcarvalo/tc-en-voice-agent-v1"


def main() -> None:
    count = sum(1 for _ in JSONL.open(encoding="utf-8"))
    wavs = len(list(AUDIO.glob("*.wav")))
    print(f"{count} dialogues, {wavs} WAVs")

    subprocess.run(["tar", "-czf", str(TARBALL), "-C", str(AUDIO.parent), AUDIO.name], check=True)
    print(f"tarball: {TARBALL.stat().st_size / 1e6:.0f} MB")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for local, remote in ((str(JSONL), "phase_b/train.jsonl"), (str(TARBALL), "phase_b/audio.tar.gz")):
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO, repo_type="dataset")
        print(f"pushed {remote}")


if __name__ == "__main__":
    main()
