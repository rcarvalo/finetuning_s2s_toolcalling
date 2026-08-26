"""Score an adapter on the uncontaminated fresh set, exactly as v2 was scored.

The fresh set (300 cases, 225 positives) is the strongest generalization
probe: unseen wording AND an unseen TTS engine. Comparability is the whole
point, so this mirrors the v2 run recorded in reports/fresh/fresh_v2.json —
same question set, same `--tool-definitions en`, same tool_call scorer.

The split is rehydrated from the Hub parquet at the STORAGE level (bytes out,
soundfile in): decoding through the `datasets` Audio feature would drag in
torchcodec for no benefit.

    python infra/eval_fresh.py --adapter Rcarvalo/lfm25-tc-en-v3-adapter \
        --out reports/fresh/fresh_v3.json
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

REPO = "Rcarvalo/tc-en-voice-agent-v1"
OUT = Path("data/fresh_eval")
AUDIO = OUT / "audio"
CASES = OUT / "cases.jsonl"


def fresh_shards() -> list[str]:
    from huggingface_hub import HfApi

    shards = sorted(
        name
        for name in HfApi().list_repo_files(REPO, repo_type="dataset")
        if name.startswith("data/test_fresh") and name.endswith(".parquet")
    )
    if not shards:
        raise FileNotFoundError(f"no test_fresh parquet in {REPO}")
    return shards


def rehydrate_fresh() -> int:
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    from lfm2_audio.data_prep.hf_rehydrate import row_to_dialogue

    AUDIO.mkdir(parents=True, exist_ok=True)
    written = 0
    with CASES.open("w", encoding="utf-8") as handle:
        for shard in fresh_shards():
            table = pq.read_table(hf_hub_download(REPO, shard, repo_type="dataset"))
            for row in table.to_pylist():
                relative = row["audio"]["path"]
                target = AUDIO / relative
                if not target.exists():
                    data, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                    sf.write(str(target), data, rate, subtype="PCM_16")
                handle.write(json.dumps(row_to_dialogue(row, relative), ensure_ascii=False) + "\n")
                written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Hub repo id or local adapter directory")
    parser.add_argument("--out", default="reports/fresh/fresh_v3.json")
    parser.add_argument("--checkpoint", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument(
        "--push-to",
        default=None,
        help="dataset repo id: upload the report as soon as it exists. A reclaimed "
        "Colab VM otherwise takes the only copy of the result with it.",
    )
    parser.add_argument(
        "--scorers",
        default="tool_call",
        help="comma-separated. 'tool_call' alone keeps the run comparable to v2/v3. "
        "Add wer,dnsmos,nisqa to also grade the generated speech (this campaign "
        "does produce audio, unlike the in-training eval), and reasoning for "
        "relevance/grounding/honesty via the LLM judge.",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"fresh set: {rehydrate_fresh()} cases rehydrated", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "lfm2_audio.cli.eval.suite",
            "--checkpoint",
            args.checkpoint,
            "--adapter",
            args.adapter,
            "--backend",
            "liquid",
            "--questions",
            str(CASES),
            "--audio-root",
            str(AUDIO),
            "--tool-definitions",
            "en",
            "--scorers",
            args.scorers,
            "--fail-on-unavailable",
            "--out",
            args.out,
        ],
        check=False,
    ).returncode
    if rc != 0:
        raise SystemExit(rc)

    report = json.loads(Path(args.out).read_text(encoding="utf-8"))
    for metric in report["metrics"]:
        # measured/skipped/unavailable, pas seulement la moyenne : un scorer qui
        # saute tous les cas rendrait sinon une moyenne d'apparence honorable.
        print(
            f"{metric['scorer']:10s} mean={metric['mean']:.4f} measured={metric['measured']} "
            f"skipped={metric['skipped']} unavailable={metric['unavailable']}",
            flush=True,
        )

    if args.push_to:
        import os

        from huggingface_hub import HfApi

        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=args.out,
            path_in_repo=f"reports/{Path(args.out).name}",
            repo_id=args.push_to,
            repo_type="dataset",
        )
        print(f"report pushed to {args.push_to}", flush=True)
    print("EVAL_FRESH_DONE", flush=True)


if __name__ == "__main__":
    main()
