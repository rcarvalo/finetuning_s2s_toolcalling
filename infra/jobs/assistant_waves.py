#!/usr/bin/env python3
"""Both assistant waves on one pod: French families into brick A, English into D.

One voice (Voxtral ``fr_female``, the identity brick A and D already carry)
over two bricks, because the mixer sizes the English share by brick. Each wave
is a separate ``build_brick_a`` process — its configuration lives in env vars
read at import — and each is resumable from the Hub on its own, so a pod lost
mid-wave costs one push interval.

    LFM2_JOB=assistant_waves   (HF_TOKEN required; BRICK_A_* overrides honoured)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

JOBS_DIR = Path(__file__).resolve().parent
V2 = "corpus/C_dialogues/dialogues_v2.jsonl"
WAVES: list[dict[str, str]] = [
    {
        "name": "fr",
        "BRICK_A_HF_PATH": "A_assistant_speech",
        "BRICK_A_SKIP_KINDS": "en",
        "BRICK_A_SOURCES": f"corpus/C_dialogues/dialogues.jsonl,corpus/TC_fr/tc_fr_v1.jsonl,{V2}",
    },
    {"name": "en", "BRICK_A_HF_PATH": "D_english", "BRICK_A_KINDS": "en", "BRICK_A_SOURCES": V2},
]
DEFAULTS = {
    "BRICK_A_ENGINE": "voxtral",
    "BRICK_A_VOICE": "fr_female",
    "BRICK_A_BATCH": "32",
    "BRICK_A_CONCURRENCY": "16",
    "BRICK_A_PUSH_EVERY": "5",
}


def wave_env(base: dict[str, str], wave: dict[str, str]) -> dict[str, str]:
    """Defaults, then the operator's env, then what the wave imposes."""
    imposed = {k: v for k, v in wave.items() if k != "name"}
    return {**DEFAULTS, **base, **imposed}


def main() -> None:
    statuses: dict[str, int] = {}
    for wave in WAVES:
        print(f"=== vague {wave['name']} → {wave['BRICK_A_HF_PATH']}", flush=True)
        result = subprocess.run(
            [sys.executable, "-u", str(JOBS_DIR / "build_brick_a.py")],
            env=wave_env(dict(os.environ), wave),
            check=False,
        )
        statuses[wave["name"]] = result.returncode
        print(f"===RESULT=== wave={wave['name']} status={result.returncode}", flush=True)
    print("===RESULT assistant_waves===", flush=True)
    print(json.dumps(statuses), flush=True)
    raise SystemExit(0 if all(code == 0 for code in statuses.values()) else 1)


if __name__ == "__main__":
    main()
