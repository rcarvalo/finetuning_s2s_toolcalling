#!/usr/bin/env python3
"""Diagnostic audio-in (sans WebRTC) : l'encodeur multimodal est-il appelé ?

Charge l'engine, vérifie si l'architecture est vue comme MULTIMODALE, puis
soumet 2 s d'audio synthétique (le CONTENU importe peu — on teste la
PLOMBERIE). Avec LFM2_DEBUG_MM=1, on doit voir les lignes ``[MM]`` :
- ``embed_multimodal`` appelé → encodeur audio actif ;
- ``embed_input_ids … is_mm.sum=N scatté=N`` → embeddings injectés.

Aucune ligne ``[MM]`` = le multimodal n'est pas branché (registre/ordre) →
l'audio est ignoré (réponses génériques). C'est le test décisif.

    python scripts/debug_audio_in.py --checkpoint /content/lfm25_audio_omni
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("LFM2_DEBUG_MM", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/content/lfm25_audio_omni")
    ap.add_argument("--no-deploy-config", action="store_true")
    args = ap.parse_args()

    from s2s_demo import VllmBackend

    deploy = None if args.no_deploy_config else REPO / "configs/vllm_omni_lfm2_audio.yaml"
    backend = VllmBackend(args.checkpoint, deploy_config=deploy)

    # 1) la classe a-t-elle bien enregistré son processor multimodal ? NB : cette
    # introspection est PROCESS-LOCAL. Le modèle tourne dans un sous-process
    # (StageEngineCoreProc) ; un False ICI ne prouve RIEN. Le signal qui FAIT FOI
    # est la ligne « [MM] embed_multimodal … » émise par ce sous-process (= le
    # conformer a bien été appelé sur l'audio). On l'imprime juste pour mémoire.
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm_omni_lfm2_audio.lfm2_audio import Lfm2AudioOmniForConditionalGeneration

        klass = Lfm2AudioOmniForConditionalGeneration
        reg = MULTIMODAL_REGISTRY
        in_mm_registry = any(
            klass in getattr(reg, attr, {})
            for attr in ("_processor_factories", "_processing_info", "_processors")
        )
        print(f"\n[CHECK] processor enregistré (process courant) : {in_mm_registry} "
              "— NON décisif ; voir les lignes « [MM] embed_multimodal » du "
              "sous-process engine (= encodeur audio réellement appelé).", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[CHECK] introspection registre impossible : {e}", flush=True)

    # 2) tour audio-in synthétique (2 s, 16 kHz) — on observe les lignes [MM]
    sr = 16_000
    wav = (0.1 * np.sin(2 * np.pi * 220 * np.arange(2 * sr) / sr)).astype(np.float32)
    print("\n[RUN] tour audio-in synthétique — chercher les lignes « [MM] » ci-dessous", flush=True)
    txt, out, m = backend.reply(audio=(wav, sr))
    print("\n========== RÉSULTAT ==========", flush=True)
    print("réponse texte :", repr(txt))
    print("frames audio générées (stage 0) :", m.get("frames"))
    print("→ si AUCUNE ligne [MM] n'est apparue : l'encodeur audio n'est pas appelé "
          "(plomberie multimodale morte).", flush=True)


if __name__ == "__main__":
    main()
