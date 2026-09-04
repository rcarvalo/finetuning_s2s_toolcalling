#!/usr/bin/env python3
"""Generate French tool-calling dialogues with Gemini and ship them to the corpus Hub.

The 150 h plan (configs/corpus/fr_150h.yaml) wants 2 500 French tool-calling
dialogues — the one brick the corpus had none of until `--lang fr` landed on
the generator. Text only, no GPU: the CPU runtime is enough. Brick A then
voices them with the assistant voice, reading the file from the same Hub path
(`build_brick_a` resolves any relative source against
`Rcarvalo/lfm25-fr-corpus-v1`).

    LFM2_JOB=generate_tc_fr LFM2_ARGS="--n-total 2500 --output corpus/TC_fr/tc_fr_v2.jsonl"

Tools and argument values stay English — a call is structural, not
linguistic; only what is spoken changes language. Every case is verified by
the parser and the registry before it is written, and the output is pushed as
soon as the generator returns, so a lost VM costs the run, never the file.

Resumable: an existing output file is uploaded as-is when `--n-total` is
already met, and the generator itself flushes each accepted case to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/content/finetuning_s2s_toolcalling"))
CORPUS_REPO = os.environ.get("LFM2_CORPUS_REPO", "Rcarvalo/lfm25-fr-corpus-v1")


def run(cmd: list[str]) -> None:
    print("===", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_deps() -> None:
    # The generator imports the Anthropic SDK and the live search backends at
    # module level; none of them sits in the extras the entrypoint installs.
    run([sys.executable, "-m", "pip", "install", "-q", "anthropic>=0.40", "ddgs>=6.0", "tavily-python>=0.5"])


def count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def push(path: Path) -> None:
    from huggingface_hub import HfApi

    relative = path.relative_to(ROOT) if path.is_absolute() else path
    HfApi().upload_file(
        path_or_fileobj=str(ROOT / relative),
        path_in_repo=str(relative),
        repo_id=CORPUS_REPO,
        repo_type="dataset",
    )
    print(f"pushed {relative} -> {CORPUS_REPO}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-total", type=int, default=2500)
    parser.add_argument("--output", default="corpus/TC_fr/tc_fr_v2.jsonl")
    parser.add_argument("--per-cell", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=6, help="lower it on 429s")
    parser.add_argument("--held-out", default=None, help="benchmark JSONL to keep out (contamination filter)")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY manquant")

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    already = count(output)
    if already >= args.n_total:
        print(f"{output} already holds {already} dialogues — uploading as is", flush=True)
    else:
        ensure_deps()
        cmd = [
            sys.executable,
            "-m",
            "lfm2_audio.cli.data.generate",
            "--output",
            str(output),
            "--lang",
            "fr",
            "--mode",
            "loop",
            "--provider",
            "gemini",
            "--n-total",
            str(args.n_total),
            "--per-cell",
            str(args.per_cell),
            "--concurrency",
            str(args.concurrency),
        ]
        if args.held_out:
            cmd += ["--held-out", args.held_out]
        run(cmd)

    total = count(output)
    langs = {}
    for line in output.read_text(encoding="utf-8").splitlines():
        if line.strip():
            meta = json.loads(line).get("meta") or {}
            langs[meta.get("lang", "?")] = langs.get(meta.get("lang", "?"), 0) + 1
    push(output)
    print(
        f"===RESULT=== tc_fr output={args.output} dialogues={total} by_lang={json.dumps(langs)} repo={CORPUS_REPO}",
        flush=True,
    )


if __name__ == "__main__":
    main()
