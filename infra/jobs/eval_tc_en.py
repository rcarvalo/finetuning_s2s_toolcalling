#!/usr/bin/env python3
"""Score an EN tool-calling adapter the way v4 and v5 were scored, in one cell.

Two measurements, deliberately both: the fresh set grades tool CHOICE (v4 and
v5 both sit at 0.830 there) and the scenarios grade the ANSWER — which is where
v5 failed its gate. Same question set, same rubric v2, same judge, so the
numbers land in the same table as docs/v4_report.md and docs/v5_report.md.

    LFM2_JOB=eval_tc_en LFM2_ARGS="--adapter Rcarvalo/lfm25-tc-en-v5_1-adapter --tag v5_1"

Needs GEMINI_API_KEY for the judge; the search backend (ddgs) is installed by
the job itself. Every report is pushed to the Hub as soon as it exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/content/finetuning_s2s_toolcalling"))
DATASET = "Rcarvalo/tc-en-voice-agent-v1"
GATES = {"relevance": 4.5, "honesty": 4.0, "coherence": 4.5}


def run(cmd: list[str]) -> None:
    print("===", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_live_tools() -> None:
    """ddgs alone, not the `tooldata` extra.

    That extra also pulls kokoro>=0.9, which refuses Python 3.13 — Colab's
    Python — and took the whole install down with it (measured by the operator
    on 28/08). The scenarios only need the search backend.
    """
    run([sys.executable, "-m", "pip", "install", "-q", "ddgs>=6.0"])


def fetch_scenario_audio() -> None:
    from huggingface_hub import hf_hub_download

    out = ROOT / "data/audio_scenarios"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.wav")):
        return
    tarball = hf_hub_download(DATASET, "scenarios/audio_scenarios.tar.gz", repo_type="dataset")
    with tarfile.open(tarball) as archive:
        archive.extractall(out, filter="data")
    print(f"scenario audio: {len(list(out.glob('*.wav')))} clips", flush=True)


def push(path: Path, in_repo: str) -> None:
    from huggingface_hub import HfApi

    if path.is_file():
        HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=in_repo, repo_id=DATASET, repo_type="dataset")


def summarise(judged: Path) -> dict[str, float]:
    """Per-criterion means, whatever nesting the judge report uses."""
    payload = json.loads(judged.read_text(encoding="utf-8"))
    means = payload.get("means") or payload.get("summary") or {}
    if not means and isinstance(payload.get("turns"), list):
        totals: dict[str, list[float]] = {}
        for turn in payload["turns"]:
            for key, value in (turn.get("scores") or {}).items():
                totals.setdefault(key, []).append(float(value))
        means = {key: sum(values) / len(values) for key, values in totals.items() if values}
    return {key: float(value) for key, value in means.items() if isinstance(value, (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="Rcarvalo/lfm25-tc-en-v5_1-adapter")
    parser.add_argument("--tag", default="v5_1")
    parser.add_argument("--skip-scenarios", action="store_true")
    args = parser.parse_args()

    fresh = ROOT / f"reports/fresh/fresh_{args.tag}.json"
    run(
        [
            sys.executable,
            "infra/eval_fresh.py",
            "--adapter",
            args.adapter,
            "--out",
            str(fresh),
            "--scorers",
            "tool_call,reasoning",
            "--push-to",
            DATASET,
        ]
    )
    fresh_score = json.loads(fresh.read_text(encoding="utf-8")).get("tool_call", {}).get("mean")
    print(f"===RESULT=== fresh tag={args.tag} tool_call={fresh_score}", flush=True)

    if args.skip_scenarios:
        return
    sys.path.insert(0, str(ROOT / "infra" / "jobs"))
    from _gemini_preflight import preflight

    preflight()

    ensure_live_tools()
    fetch_scenario_audio()
    scen = ROOT / f"reports/scenarios_{args.tag}"
    run(
        [
            sys.executable,
            "infra/pod_scenarios.py",
            "--adapter",
            args.adapter,
            "--out-dir",
            str(scen),
            "--push-to",
            DATASET,
        ]
    )
    judged = scen / "judged.json"
    run(
        [
            sys.executable,
            "infra/judge_scenarios.py",
            "--transcript",
            str(scen / "transcript.jsonl"),
            "--out",
            str(judged),
            "--rubric",
            "v2",
            "--limit",
            "60",
        ]
    )
    push(judged, f"reports/scenarios_{args.tag}/judged.json")

    means = summarise(judged)
    verdict = {key: ("OK" if means.get(key, 0.0) >= gate else "FAIL") for key, gate in GATES.items()}
    print(f"===RESULT=== scenarios tag={args.tag} {json.dumps(means)} gates={json.dumps(verdict)}", flush=True)


if __name__ == "__main__":
    main()
