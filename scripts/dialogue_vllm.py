#!/usr/bin/env python3
"""Dialogue interactif LFM2.5-Audio via vLLM-Omni.

Initialise l'engine une fois, puis boucle sur les entrées utilisateur.
L'audio de chaque réponse est sauvegardé dans /tmp/lfm2_reply_<N>.wav
et joué via sounddevice si disponible.

Usage :
    python scripts/dialogue_vllm.py --checkpoint /content/lfm25_audio_omni
    python scripts/dialogue_vllm.py --checkpoint /content/lfm25_audio_omni \\
        --deploy-config configs/vllm_omni_lfm2_audio.yaml
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_engine(checkpoint: Path, deploy_config: Path | None, gpu_util: float, dtype: str):
    import vllm_omni.plugins as _p

    _p.omni_plugins_loaded = False
    import vllm_omni_lfm2_audio  # noqa: F401

    from vllm_omni.plugins import load_omni_general_plugins
    load_omni_general_plugins()

    from vllm_omni import Omni

    kwargs: dict = dict(
        model=str(checkpoint),
        async_chunk=True,
        stage_init_timeout=1200,
        init_timeout=1800,
    )
    if deploy_config:
        # eager/gpu_util/dtype par stage viennent du YAML (stage 0 en CUDA
        # graphs, stage 1 eager) — ne pas les écraser globalement ici.
        kwargs["deploy_config"] = str(deploy_config)
    else:
        kwargs.update(
            enforce_eager=True,
            gpu_memory_utilization=gpu_util,
            dtype=dtype,
            async_scheduling=False,  # écart 5 : async tronque l'historique du sampler
        )

    t0 = time.time()
    omni = Omni(**kwargs)
    print(f"[engine] prêt en {time.time()-t0:.0f}s — async_chunk={omni.async_chunk}", flush=True)
    return omni


def _build_renderer(checkpoint: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(checkpoint))

    def render(history: list[tuple[str, str]], system: str) -> list[int]:
        s = f"<|startoftext|><|im_start|>system\n{system}<|im_end|>\n"
        for role, txt in history:
            s += f"<|im_start|>{role}\n{txt}<|im_end|>\n"
        return tok(s + "<|im_start|>assistant\n", add_special_tokens=False).input_ids

    return render


def _extract_wave(x) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("model_outputs", "audio", "waveform", "wav"):
            if k in x:
                return _extract_wave(x[k])
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy().reshape(-1)
    if isinstance(x, np.ndarray):
        return x.reshape(-1)
    return None


def _save_wav(wav: np.ndarray, path: Path, rate: int = 24_000) -> None:
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32_767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16.tobytes())


def _play(wav: np.ndarray, rate: int = 24_000) -> None:
    try:
        import sounddevice as sd

        sd.play(wav, samplerate=rate, blocking=False)
    except Exception:
        pass  # sounddevice indisponible (Colab sans audio) — le WAV est sauvegardé


# ── génération ────────────────────────────────────────────────────────────────

def say(
    omni,
    render,
    history: list[tuple[str, str]],
    user_text: str,
    system: str,
    max_tokens: int,
    out_dir: Path,
    turn: int,
):
    from vllm import SamplingParams
    from vllm_omni_lfm2_audio.constants import (
        AUDIO_EOA_PLACEHOLDER_ID,
        AUDIO_FRAME_PLACEHOLDER_ID,
        IM_END_TOKEN_ID,
    )

    history.append(("user", user_text))
    ids = render(history, system)
    sp0 = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=[IM_END_TOKEN_ID])
    sp1 = SamplingParams(max_tokens=1, detokenize=False)

    t0 = time.time()
    outs = omni.generate({"prompt_token_ids": ids}, [sp0, sp1])
    dt = time.time() - t0

    text, wav, nf = "", None, 0
    for o in outs:
        ro = o.request_output
        if o.final_output_type == "text" and ro and ro.outputs:
            toks = list(ro.outputs[0].token_ids)
            nf = sum(1 for t in toks if t in (AUDIO_FRAME_PLACEHOLDER_ID, AUDIO_EOA_PLACEHOLDER_ID))
            text = ro.outputs[0].text
        elif o.final_output_type == "audio":
            wav = _extract_wave(
                getattr(ro, "multimodal_output", None) or getattr(o, "multimodal_output", None)
            )

    history.append(("assistant", text))

    # Vérification anti-duplication : durée attendue = nf / 12.5
    expected_s = nf / 12.5
    actual_s = wav.size / 24_000 if wav is not None and wav.size else 0
    dup_flag = ""
    if actual_s > expected_s * 1.5:
        dup_flag = f"  ⚠️  duplication détectée ({actual_s:.1f}s vs {expected_s:.1f}s attendu) — git pull ?"

    print(f"\n[{dt:.1f}s | {actual_s:.1f}s audio{dup_flag}]")
    print(f"🤖 {text}")

    wav_path = out_dir / f"reply_{turn:03d}.wav"
    if wav is not None and wav.size:
        _save_wav(wav, wav_path)
        print(f"   → {wav_path}")
        _play(wav)

    return text


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deploy-config", type=Path, default=None)
    parser.add_argument("--system", default="You are a helpful assistant. Respond with interleaved text and audio.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--gpu-util", type=float, default=0.42)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/lfm2_dialogue"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    omni = _load_engine(args.checkpoint, args.deploy_config, args.gpu_util, args.dtype)
    render = _build_renderer(args.checkpoint)
    history: list[tuple[str, str]] = []

    print("Dialogue — Ctrl+C ou entrée vide pour quitter.")
    turn = 0
    try:
        while True:
            try:
                u = input("\n👤 ").strip()
            except EOFError:
                break
            if not u:
                break
            turn += 1
            say(omni, render, history, u, args.system, args.max_tokens, args.out_dir, turn)
    except KeyboardInterrupt:
        print("\nInterrompu.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
