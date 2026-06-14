#!/usr/bin/env python3
"""Rejoue des WAV comme des tours user CONSÉCUTIFS dans le backend vLLM-Omni.

Sort WebRTC + VAD de l'équation : on alimente directement
``VllmBackend.reply(audio=…)`` tour après tour (l'historique s'accumule) et on
affiche la réponse texte + l'historique. C'est le test PROPRE du fix multi-tours
(1 placeholder = 1 audio par tour) sur des entrées connues et reproductibles.

    python scripts/multiturn_replay.py --checkpoint /content/lfm25_audio_omni \
        /tmp/lfm2_input/in_000.wav /tmp/lfm2_input/in_001.wav

Sans argument de fichier : rejoue tous les ``/tmp/lfm2_input/*.wav`` triés.
``--transcribe`` : sonde ASR (affiche ce que le modèle ENTEND, mot-à-mot).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def _load_16k(path: Path) -> tuple[np.ndarray, int]:
    wave, sr = sf.read(str(path), dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    rms = float(np.sqrt(np.mean(wave**2))) if wave.size else 0.0
    print(f"  · {path.name} : {wave.size/sr:.2f}s @ {sr}Hz · RMS {rms:.3f}", flush=True)
    return wave, sr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/content/lfm25_audio_omni")
    ap.add_argument("--no-deploy-config", action="store_true")
    ap.add_argument("--transcribe", action="store_true",
                    help="sonde ASR : le modèle restitue le texte de ce qu'il entend")
    ap.add_argument("wavs", nargs="*", help="WAV à rejouer en tours successifs")
    args = ap.parse_args()

    wavs = [Path(w) for w in args.wavs] or sorted(Path("/tmp/lfm2_input").glob("*.wav"))
    if not wavs:
        ap.error("aucun WAV (ni argument ni /tmp/lfm2_input/*.wav)")

    from s2s_demo import VllmBackend

    deploy = None if args.no_deploy_config else REPO / "configs/vllm_omni_lfm2_audio.yaml"
    backend = VllmBackend(args.checkpoint, deploy_config=deploy)
    if args.transcribe:
        backend.system = ("Transcribe the user's speech verbatim. "
                          "Respond with the transcription only, as text.")
        print("[MODE] transcription (sonde ASR)", flush=True)

    print(f"\n{len(wavs)} tour(s) à rejouer :", flush=True)
    for i, path in enumerate(wavs):
        print(f"\n===== TOUR {i} =====", flush=True)
        wave, sr = _load_16k(path)
        txt, _, m = backend.reply(audio=(wave, sr))
        print(f"  🤖 {txt!r}  (frames audio: {m.get('frames')})", flush=True)

    print("\n===== HISTORIQUE FINAL (ce que voit le modèle) =====", flush=True)
    for role, content in backend.history:
        print(f"  [{role}] {content[:80]!r}", flush=True)


if __name__ == "__main__":
    main()
