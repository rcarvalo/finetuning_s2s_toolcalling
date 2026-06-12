#!/usr/bin/env python3
"""Bench TTFA (time-to-first-audio) — objectif « ElevenLabs » : 200-500 ms.

Mesure, en streaming async_chunk (py_generator), pour chaque requête :
  - TTFT  : premier token de texte décodé par le stage 0 ;
  - TTFA  : premier chunk audio DÉCODÉ reçu du stage 1 (ce que l'utilisateur
            entend) — c'est la métrique comparable à ElevenLabs ;
  - total : fin de génération ; RTF = total / durée d'audio produite.

Usage (pod GPU) :
    python scripts/bench_ttfa.py --checkpoint /workspace/models/lfm25_audio_omni \\
        --deploy-config configs/vllm_omni_lfm2_audio.yaml --runs 5

A/B utiles :
  - eager vs CUDA graphs : éditer enforce_eager du stage 0 dans le YAML ;
  - chunk initial : initial_codec_chunk_frames (2 par défaut) vs sans (=10).
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

PROMPTS = [
    "Bonjour, qui es-tu ?",
    "Quelle heure est-il à Paris ?",
    "Raconte-moi une courte histoire.",
]


def _load_engine(checkpoint: Path, deploy_config: Path | None):
    try:
        import vllm  # noqa: F401
    except ImportError:
        raise SystemExit(
            "vllm n'est pas installé — vllm-omni ne le déclare pas en dépendance "
            "(versions appariées major.minor). Sur CUDA 12 (Colab), le build PyPI "
            "(CUDA 13) ne marche pas, utiliser le wheel +cu129 du release :\n"
            "    pip install 'vllm @ https://github.com/vllm-project/vllm/releases/"
            "download/v0.22.1/vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl' "
            "--extra-index-url https://download.pytorch.org/whl/cu129"
        ) from None

    import vllm_omni.plugins as _p

    _p.omni_plugins_loaded = False
    import vllm_omni_lfm2_audio  # noqa: F401

    from vllm_omni.plugins import load_omni_general_plugins
    load_omni_general_plugins()

    from vllm_omni import Omni

    kwargs: dict = dict(model=str(checkpoint), async_chunk=True,
                        stage_init_timeout=1200, init_timeout=1800)
    if deploy_config:
        kwargs["deploy_config"] = str(deploy_config)
    else:
        kwargs.update(enforce_eager=True, gpu_memory_utilization=0.42,
                      dtype="bfloat16", async_scheduling=False)
    t0 = time.time()
    omni = Omni(**kwargs)
    print(f"[engine] prêt en {time.time()-t0:.0f}s — async_chunk={omni.async_chunk}", flush=True)
    return omni


def _wave(x) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("model_outputs", "audio", "waveform", "wav"):
            if k in x:
                return _wave(x[k])
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy().reshape(-1)
    if isinstance(x, np.ndarray):
        return x.reshape(-1)
    return None


def bench_once(omni, prompt_ids: list[int], max_tokens: int) -> dict:
    from vllm import SamplingParams
    from vllm_omni_lfm2_audio.constants import IM_END_TOKEN_ID

    sp = [
        SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=[IM_END_TOKEN_ID]),
        SamplingParams(max_tokens=1, detokenize=False),
    ]
    t0 = time.time()
    ttft = ttfa = None
    n_chunks, samples = 0, 0
    for out in omni.generate({"prompt_token_ids": prompt_ids}, sp, py_generator=True):
        now = time.time()
        if out.final_output_type == "text" and ttft is None:
            ro = out.request_output
            if ro and ro.outputs and (ro.outputs[0].token_ids or ro.outputs[0].text):
                ttft = now - t0
        elif out.final_output_type == "audio":
            ro = out.request_output
            w = _wave(getattr(out, "multimodal_output", None) or getattr(ro, "multimodal_output", None))
            if w is not None and w.size:
                if ttfa is None:
                    ttfa = now - t0
                n_chunks += 1
                samples += int(w.size)
    total = time.time() - t0
    audio_s = samples / 24_000
    return {"ttft": ttft, "ttfa": ttfa, "total": total, "audio_s": audio_s,
            "chunks": n_chunks, "rtf": total / audio_s if audio_s else float("inf")}


def _fmt_ms(v: float | None) -> str:
    return f"{v*1000:7.0f} ms" if v is not None else "      —   "


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--deploy-config", type=Path, default=Path("configs/vllm_omni_lfm2_audio.yaml"))
    ap.add_argument("--no-deploy-config", action="store_true", help="kwargs legacy (eager) au lieu du YAML")
    ap.add_argument("--runs", type=int, default=5, help="runs mesurés par prompt")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--system", default="You are a helpful assistant. Respond with interleaved text and audio.")
    args = ap.parse_args()

    omni = _load_engine(args.checkpoint, None if args.no_deploy_config else args.deploy_config)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.checkpoint))

    def render(user: str) -> list[int]:
        s = (f"<|startoftext|><|im_start|>system\n{args.system}<|im_end|>\n"
             f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        return tok(s, add_special_tokens=False).input_ids

    for i in range(args.warmup):
        r = bench_once(omni, render(PROMPTS[i % len(PROMPTS)]), args.max_tokens)
        print(f"[warmup {i+1}] ttfa={_fmt_ms(r['ttfa'])} total={r['total']:.1f}s")

    rows = []
    for i in range(args.runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        r = bench_once(omni, render(prompt), args.max_tokens)
        rows.append(r)
        print(f"[run {i+1}] ttft={_fmt_ms(r['ttft'])} ttfa={_fmt_ms(r['ttfa'])} "
              f"total={r['total']:5.1f}s audio={r['audio_s']:5.1f}s "
              f"chunks={r['chunks']:3d} rtf={r['rtf']:.2f}  « {prompt[:30]} »")

    ttfas = [r["ttfa"] for r in rows if r["ttfa"] is not None]
    if not ttfas:
        print("\n❌ aucun chunk audio reçu — vérifier async_chunk / initial_codec_chunk_frames")
        return 1
    p50 = statistics.median(ttfas)
    p95 = max(ttfas) if len(ttfas) < 20 else statistics.quantiles(ttfas, n=20)[18]
    rtfs = [r["rtf"] for r in rows if r["audio_s"]]
    print(f"\nTTFA p50={p50*1000:.0f} ms  p95={p95*1000:.0f} ms  "
          f"(n={len(ttfas)})  RTF médian={statistics.median(rtfs):.2f}")
    target_ok = p50 <= 0.5
    print("🎯 objectif 200-500 ms : " + ("ATTEINT ✅" if target_ok else "pas encore ❌"))
    if statistics.median(rtfs) > 1.0:
        print("⚠️  RTF > 1 : la lecture rattrapera la génération (trous après le 1er chunk) — "
              "voir docs/optimization_audit.md §1.3-1.4 (CUDA graphs depthformer/détokeniseur)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
