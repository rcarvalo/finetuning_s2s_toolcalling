"""Assistant-turn TTS via a Qwen3-TTS server on Colab (vLLM-Omni speech API).

The LATEST vllm/vllm-omni pair serves Qwen3-TTS behind the OpenAI-compatible
``/v1/audio/speech`` endpoint — the exact protocol our synthesize CLI already
speaks (its "voxtral" engine is a generic OpenAI-speech client). So this job
is: serve, then run the existing CLI at concurrency 8. The 0.22-era CUDA-13
wheel problem does not apply: recent vLLM ships wheels for today's Colab.

Input: the users-voiced JSONL (user turns already carry audio → the CLI skips
them and voices only the assistant answer turns, with the fixed persona).
Output: assistant WAVs, tarballed and pushed to the Hub.

Fallback if the server route fails: infra/colab_qwen_tts.py (plain qwen-tts
transformers loop, no vLLM at all).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import httpx

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
USERS_JSONL = Path("data/phase_b_train_users.jsonl")
AUDIO = Path("data/audio_phase_b_assistant")
HUB_REPO = "Rcarvalo/tc-en-voice-agent-v1"
SPEAKER = os.environ.get("ASSISTANT_VOICE", "Aiden")


def wait_healthy(timeout_s: int = 900) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get("http://localhost:8000/health", timeout=5).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(5)
    return False


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)

    if not USERS_JSONL.exists():
        from huggingface_hub import hf_hub_download

        USERS_JSONL.parent.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(HUB_REPO, "phase_b/train_users.jsonl", repo_type="dataset")
        USERS_JSONL.write_bytes(Path(path).read_bytes())

    server = subprocess.Popen(
        ["vllm", "serve", MODEL_ID, "--omni", "--gpu-memory-utilization", "0.85"],
        stdout=open("/content/qwen_tts_server.log", "w"),  # noqa: SIM115 — outlives this scope
        stderr=subprocess.STDOUT,
    )
    print("server starting…", flush=True)
    if not wait_healthy():
        print("SERVER_DEAD — full log follows", flush=True)
        print(Path("/content/qwen_tts_server.log").read_text()[-4000:], flush=True)
        raise SystemExit(1)
    print("server healthy", flush=True)

    # User turns already carry audio in this JSONL → only assistant answer
    # turns get voiced, with the fixed persona (stable voice = v3's voice).
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "lfm2_audio.cli.data.synthesize",
            "--engine",
            "voxtral",  # generic OpenAI-speech client
            "--split",
            "train",
            "--voices",
            SPEAKER,
            "--assistant-voice",
            SPEAKER,
            "--dialogues",
            str(USERS_JSONL),
            "--audio-root",
            str(AUDIO),
            "--out",
            "data/phase_b_train.jsonl",
            "--concurrency",
            "8",
        ],
        check=False,
    ).returncode
    print(f"synthesis rc={rc}", flush=True)
    server.terminate()
    if rc != 0:
        raise SystemExit(rc)

    tarball = "/tmp/phase_b_assistant_audio.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(str(AUDIO), arcname=AUDIO.name)
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj=tarball, path_in_repo="phase_b/assistant_audio.tar.gz", repo_id=HUB_REPO, repo_type="dataset"
    )
    api.upload_file(
        path_or_fileobj="data/phase_b_train.jsonl",
        path_in_repo="phase_b/train.jsonl",
        repo_id=HUB_REPO,
        repo_type="dataset",
    )
    print("QWEN_TTS_DONE", flush=True)


if __name__ == "__main__":
    main()
