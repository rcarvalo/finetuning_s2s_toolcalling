"""Synthesize the fresh test set through a local Voxtral server (RunPod pod).

Reads ``data/test_fresh_src.jsonl`` (300 unseen utterances, committed), speaks
every user turn with the HELD-OUT voices, writes WAVs + the final JSONL, then
pushes the ``test_fresh`` split (parquet, audio bytes) to the curated dataset
repo. Resumable: existing WAVs are kept, not re-synthesized.

Runs inside the pod only — the server lives at localhost:8000.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, "/repo/python")
from lfm2_audio.data_prep.turn_sampling import turn_random

SERVER = "http://localhost:8000/v1/audio/speech"
MODEL = "mistralai/Voxtral-4B-TTS-2603"
HELD_OUT_VOICES = ["neutral_male", "neutral_female"]
SEED = 42

SRC = Path("/repo/data/test_fresh_src.jsonl")
AUDIO_ROOT = Path("/repo/data/audio_test_fresh")
OUT = Path("/repo/data/test_fresh.jsonl")


def synthesize(client: httpx.Client, text: str, voice: str, target: Path) -> None:
    response = client.post(
        SERVER,
        json={"input": text, "model": MODEL, "response_format": "wav", "voice": voice},
        timeout=180,
    )
    response.raise_for_status()
    target.write_bytes(response.content)


def main() -> None:
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    dialogues = [json.loads(line) for line in SRC.open(encoding="utf-8")]

    jobs: list[tuple[dict[str, Any], dict[str, Any], str, str]] = []
    reused = 0
    for dialogue in dialogues:
        for index, turn in enumerate(dialogue["turns"]):
            if turn.get("role") != "user" or not turn.get("text"):
                continue
            voice = HELD_OUT_VOICES[turn_random(SEED, dialogue["id"], index).randrange(len(HELD_OUT_VOICES))]
            rel = f"{dialogue['id']}_u{index}.wav"
            turn["voice"] = voice
            if (AUDIO_ROOT / rel).exists():
                turn["audio"] = rel
                reused += 1
                continue
            jobs.append((dialogue, turn, voice, rel))
    print(f"{len(jobs)} utterances to speak ({reused} reused)", flush=True)

    failures = 0
    with httpx.Client() as client:

        def run(job: tuple[dict[str, Any], dict[str, Any], str, str]) -> bool:
            dialogue, turn, voice, rel = job
            try:
                synthesize(client, turn["text"], voice, AUDIO_ROOT / rel)
            except Exception as exc:  # one bad utterance must not sink the batch
                print(f"[fail] {dialogue['id']}: {exc}", flush=True)
                return False
            turn["audio"] = rel
            return True

        with ThreadPoolExecutor(max_workers=8) as pool:
            for done, ok in enumerate(pool.map(run, jobs), 1):
                failures += not ok
                if done % 50 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} ({failures} failures)", flush=True)

    spoken = [
        d for d in dialogues if all(t.get("audio") for t in d["turns"] if t.get("role") == "user" and t.get("text"))
    ]
    with OUT.open("w", encoding="utf-8") as handle:
        for dialogue in spoken:
            handle.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
    print(f"{len(spoken)}/{len(dialogues)} dialogues fully spoken → {OUT}", flush=True)

    _push_split(spoken)
    print("TTS_JOB_DONE", flush=True)


def _push_split(dialogues: list[dict[str, Any]]) -> None:
    """Upload the split in the same flat layout as the other test splits."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    rows = []
    for dialogue in dialogues:
        user = next(t for t in dialogue["turns"] if t["role"] == "user")
        calls = next((t.get("tool_calls") for t in dialogue["turns"] if t["role"] == "assistant"), None)
        call = (calls or [None])[0]
        meta = dialogue.get("meta") or {}
        rows.append(
            {
                "id": dialogue["id"],
                "audio": {"bytes": (AUDIO_ROOT / user["audio"]).read_bytes(), "path": user["audio"]},
                "utterance": user.get("text", ""),
                "has_tool_call": bool(call),
                "tool_name": (call or {}).get("name"),
                "arguments": json.dumps((call or {}).get("arguments")) if call else None,
                "assistant_text": next(
                    (t.get("text") for t in dialogue["turns"] if t["role"] == "assistant" and t.get("text")), None
                ),
                "expected_calls": json.dumps(calls or []),
                "voice": user.get("voice"),
                "target": meta.get("target"),
                "style": meta.get("style"),
                "depth": meta.get("depth"),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), "/repo/test_fresh.parquet")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj="/repo/test_fresh.parquet",
        path_in_repo="data/test_fresh.parquet",
        repo_id="Rcarvalo/tc-en-voice-agent-v1",
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(OUT),
        path_in_repo="data/test_fresh.jsonl",
        repo_id="Rcarvalo/tc-en-voice-agent-v1",
        repo_type="dataset",
    )
    print(f"pushed {len(rows)} fresh cases to the Hub", flush=True)


if __name__ == "__main__":
    main()
