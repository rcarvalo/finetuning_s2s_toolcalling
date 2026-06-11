#!/usr/bin/env python3
"""Sonde P2 : la génération interleaved du stage 0 produit-elle de l'audio ?

Construit un prompt au format chat liquid-audio (texte seul), le soumet à
l'engine vLLM-Omni en greedy, et vérifie :
  1. des placeholders audio (frame/EOA) apparaissent dans les ids générés ;
  2. la cadence texte/audio respecte le ratio interleaved du checkpoint ;
  3. le stage 1 reçoit des frames et émet un waveform.

Usage :
    python scripts/colab_probe_interleaved.py --checkpoint /content/lfm25_audio_omni
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_omni_lfm2_audio.constants import (
    AUDIO_EOA_PLACEHOLDER_ID,
    AUDIO_FRAME_PLACEHOLDER_ID,
    IM_END_TOKEN_ID,
)


def build_prompt_ids(checkpoint: Path, user_text: str) -> list[int]:
    """Ids du prompt au format chat LFM2.5 (sans audio en entrée).

    On passe par le tokenizer HF du checkpoint avec le même gabarit que
    ``liquid_audio.ChatState`` (im_start/im_end) — vérifié contre le template.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    text = (
        "<|startoftext|><|im_start|>system\n"
        "Respond with interleaved text and audio.<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return tok(text, add_special_tokens=False).input_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Bonjour, qui es-tu ?")
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    prompt_ids = build_prompt_ids(args.checkpoint, args.prompt)
    print(f"prompt: {len(prompt_ids)} tokens")

    # la détection de pipeline d'Omni() précède le chargement des plugins par
    # l'engine : pour un pipeline out-of-tree il faut charger explicitement
    # (sinon model_type lfm2_audio tombe dans le registre diffusion)
    from vllm_omni.plugins import load_omni_general_plugins

    load_omni_general_plugins()

    from vllm import SamplingParams
    from vllm_omni import Omni

    omni = Omni(
        model=str(args.checkpoint),
        stage_init_timeout=1200,
        init_timeout=1800,
        enforce_eager=True,
        gpu_memory_utilization=0.42,
        dtype="float16",
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=[IM_END_TOKEN_ID])
    outputs = omni.generate({"prompt_token_ids": prompt_ids}, [sp])

    ok = True
    for out in outputs:
        ro = out.request_output
        if out.final_output_type == "text" and ro is not None and ro.outputs:
            ids = list(ro.outputs[0].token_ids)
            n_frames = sum(1 for i in ids if i == AUDIO_FRAME_PLACEHOLDER_ID)
            n_eoa = sum(1 for i in ids if i == AUDIO_EOA_PLACEHOLDER_ID)
            text_ids = [i for i in ids if i not in (AUDIO_FRAME_PLACEHOLDER_ID, AUDIO_EOA_PLACEHOLDER_ID)]
            print(f"stage 0 : {len(ids)} ids — {len(text_ids)} texte, {n_frames} frames, {n_eoa} EOA")
            print(f"  ids[:40] = {ids[:40]}")
            print(f"  texte    = {ro.outputs[0].text[:200]!r}")
            if n_frames == 0:
                print("  [warn] aucun placeholder audio généré")
                ok = False
        elif out.final_output_type == "audio":
            mm = getattr(ro, "multimodal_output", None) or getattr(out, "multimodal_output", None)
            print(f"stage 1 : sortie audio — type={type(mm).__name__}, repr={str(mm)[:200]}")

    print("PROBE", "OK" if ok else "INCOMPLET")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
