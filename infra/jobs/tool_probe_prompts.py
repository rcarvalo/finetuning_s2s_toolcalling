#!/usr/bin/env python3
"""Does a wording change free the model to pick a tool it never saw? Prompt-only probe.

The 28/08 probe (docs/tool_generalization_probe.md) found the model reads the
declared tool list — zero hallucination — but never picks `delegate` when a
trained tool is available (0/4). The trained instruction sentence names its two
tools and orders "call at most one", and that is a plausible cause on its own.
This job holds everything else fixed and swaps only that sentence:

    trained            the sentence the adapters were trained with
    generic            "pick the tool whose description matches"
    generic_delegate   generic + explicit precedence of delegate on multi-step asks

over the two conditions where it matters (2plus1: delegate beside the trained
tools; unseen_only: no trained tool at all), on each adapter requested.

    LFM2_JOB=tool_probe_prompts LFM2_ARGS="--adapters v4,v5_1"

Reports and per-sample archives go to the Hub; the breakdown printed between
===RESULT=== markers is the verdict: unseen-tool routing per prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/content/finetuning_s2s_toolcalling"))
DATASET = "Rcarvalo/tc-en-voice-agent-v1"
PROBE = ROOT / "benchmark/tool_probe"
REPORTS = ROOT / "reports/tool_probe_prompts"
CONDITIONS = ("2plus1", "unseen_only")
PROMPTS = {"trained": None, "generic": "generic.txt", "generic_delegate": "generic_delegate.txt"}
TRAINED = {"web_search", "db_query"}
CALL = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")


def fetch_probe() -> None:
    from huggingface_hub import hf_hub_download

    (PROBE / "audio").mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        for kind in ("tools_%s.json", "cases_%s.jsonl"):
            name = kind % cond
            src = hf_hub_download(DATASET, f"tool_probe/cases/{name}", repo_type="dataset")
            (PROBE / name).write_bytes(Path(src).read_bytes())
    if not any((PROBE / "audio").glob("*.wav")):
        tarball = hf_hub_download(DATASET, "tool_probe/audio.tar.gz", repo_type="dataset")
        with tarfile.open(tarball) as archive:
            archive.extractall(PROBE / "audio", filter="data")
    print(f"probe: {len(list((PROBE / 'audio').glob('*.wav')))} clips", flush=True)


def evaluate(adapter: str, cond: str, prompt: str) -> Path:
    archive = REPORTS / f"{adapter}_{cond}_{prompt}_samples"
    report = REPORTS / f"{adapter}_{cond}_{prompt}.json"
    cmd = [
        sys.executable,
        "-m",
        "lfm2_audio.cli.eval.suite",
        "--checkpoint",
        "LiquidAI/LFM2.5-Audio-1.5B",
        "--adapter",
        f"Rcarvalo/lfm25-tc-en-{adapter}-adapter",
        "--backend",
        "liquid",
        "--questions",
        str(PROBE / f"cases_{cond}.jsonl"),
        "--audio-root",
        str(PROBE / "audio"),
        "--tool-definitions",
        str(PROBE / f"tools_{cond}.json"),
        "--scorers",
        "tool_call",
        "--archive",
        str(archive),
        "--out",
        str(report),
    ]
    if PROMPTS[prompt]:
        cmd += ["--system-instructions", str(PROBE / "prompts" / PROMPTS[prompt])]
    print("===", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return archive


def called(text: str) -> list[str]:
    block = (
        text.split("<|tool_call_start|>", 1)[1].split("<|tool_call_end|>", 1)[0]
        if "<|tool_call_start|>" in text
        else text
    )
    return CALL.findall(block)


def breakdown(archive: Path, cond: str) -> dict[str, str]:
    declared = {t["name"] for t in json.loads((PROBE / f"tools_{cond}.json").read_text(encoding="utf-8"))}
    buckets: dict[str, list[int]] = defaultdict(list)
    hallucinated = total = 0
    for sample in sorted(archive.glob("*.json")):
        row = json.loads(sample.read_text(encoding="utf-8"))
        expected = row.get("expected_calls") or [{}]
        want = expected[0].get("name") if expected and expected[0] else None
        got = called(row.get("predicted_text") or "")
        total += 1
        hallucinated += any(name not in declared for name in got)
        ok = int(bool(got) and got[0] == want) if want else int(not got)
        buckets["neg" if want is None else ("trained" if want in TRAINED else "unseen")].append(ok)

    def rate(key: str) -> str:
        values = buckets.get(key, [])
        return f"{sum(values)}/{len(values)}" if values else "-"

    return {
        "unseen": rate("unseen"),
        "trained": rate("trained"),
        "neg": rate("neg"),
        "halluc": f"{hallucinated}/{total}",
    }


def push(archive: Path, report: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    if report.is_file():
        api.upload_file(
            path_or_fileobj=str(report),
            path_in_repo=f"tool_probe/prompts/{report.name}",
            repo_id=DATASET,
            repo_type="dataset",
        )
    if archive.is_dir():
        tarball = archive.with_suffix(".tar.gz")
        with tarfile.open(tarball, "w:gz") as tar:
            for item in sorted(archive.glob("*")):
                tar.add(item, arcname=item.name)
        api.upload_file(
            path_or_fileobj=str(tarball),
            path_in_repo=f"tool_probe/prompts/{tarball.name}",
            repo_id=DATASET,
            repo_type="dataset",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters", default="v4,v5_1", help="comma-separated adapter tags")
    args = parser.parse_args()

    fetch_probe()
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for adapter in [a.strip() for a in args.adapters.split(",") if a.strip()]:
        for cond in CONDITIONS:
            for prompt in PROMPTS:
                archive = evaluate(adapter, cond, prompt)
                stats = breakdown(archive, cond)
                push(archive, REPORTS / f"{adapter}_{cond}_{prompt}.json")
                rows.append(
                    f"{adapter:5} {cond:12} {prompt:17} unseen={stats['unseen']:6} "
                    f"trained={stats['trained']:6} neg={stats['neg']:4} halluc={stats['halluc']}"
                )
                print("===RESULT=== " + rows[-1], flush=True)
    print("===RESULT=== TABLE\n" + "\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
