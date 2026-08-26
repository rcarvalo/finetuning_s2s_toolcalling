"""Bench de latence du serving — objectif TTFA 200-500 ms.

    lfm2-bench --checkpoint exports/lfm25_audio_fr_omni --runs 5

A/B utiles : ``enforce_eager`` du stage 0 dans le YAML de déploiement (eager vs
CUDA graphs), et ``initial_codec_chunk_frames`` (2 par défaut, 10 = sans).
"""

from __future__ import annotations

import argparse
import logging
import sys

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.ds.inference_config import EngineConfig
from lfm2_audio.evaluation.latency import (
    PROMPTS_BY_LANGUAGE,
    LatencyReport,
    format_ms,
)
from lfm2_audio.evaluation.latency_benchmark import LatencyBenchmark
from lfm2_audio.serving import LFM2Audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--backend", choices=["auto", "vllm", "liquid"], default="auto")
    parser.add_argument("--runs", type=int, default=5, help="tours mesurés")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--system", default="Respond with interleaved text and audio.")
    parser.add_argument(
        "--lang",
        choices=sorted(PROMPTS_BY_LANGUAGE),
        default="en",
        help="langue des prompts de mesure (le TTFA depend du prompt : une serie par langue)",
    )
    parser.add_argument(
        "--no-deploy-config",
        action="store_true",
        help="kwargs legacy (tout eager) au lieu du YAML par stage",
    )
    return parser


def print_report(report: LatencyReport) -> None:
    for index, sample in enumerate(report.samples, start=1):
        rtf = f"{sample.real_time_factor:.2f}" if sample.real_time_factor else "—"
        print(
            f"[run {index}] ttfa={format_ms(sample.ttfa_s):>9} "
            f"total={sample.total_s:5.1f}s audio={sample.audio_s:5.1f}s "
            f"frames={sample.audio_frames:3d} rtf={rtf}"
        )
        print(f"        🤖 {sample.text[:90]!r}")

    problem = report.diagnose()
    if problem:
        print(f"\n❌ {problem}")
        return

    median_rtf = report.median_rtf
    summary = f"\nTTFA p50={format_ms(report.ttfa_p50)}  p95={format_ms(report.ttfa_p95)}  (n={len(report.measured)})"
    if median_rtf is not None:
        summary += f"  RTF médian={median_rtf:.2f}"
    print(summary)
    print("🎯 objectif 200-500 ms : " + ("ATTEINT ✅" if report.meets_target else "pas encore ❌"))
    if median_rtf and median_rtf > 1.0:
        print(
            "⚠️  RTF > 1 : la lecture rattrapera la génération (trous après le 1er chunk) "
            "— cf. docs/optimization_audit.md"
        )


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    engine = EngineConfig(deploy_config=None) if args.no_deploy_config else EngineConfig()
    try:
        model = LFM2Audio.from_pretrained(
            args.checkpoint,
            backend=args.backend,
            adapter=args.adapter,
            system=args.system,
            engine=engine,
        )
    except Lfm2AudioError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1

    with model:
        benchmark = LatencyBenchmark(model, prompts=PROMPTS_BY_LANGUAGE[args.lang], max_tokens=args.max_tokens)
        benchmark.warmup(args.warmup)
        report = benchmark.run(args.runs)

    print_report(report)
    return 0 if report.measured else 1


if __name__ == "__main__":
    raise SystemExit(main())
