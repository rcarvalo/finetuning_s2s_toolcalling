#!/usr/bin/env python3
"""Conversation vocale MAINS-LIBRES (WebRTC + VAD) avec LFM2.5-Audio / vLLM-Omni.

Tu parles, tu fais une pause → la réponse vocale démarre (~200 ms) PENDANT
qu'elle se génère, sans bouton. Audio-in NATIF : un seul engine vLLM, plus
d'ASR liquid co-chargé (l'ancienne version de ce script, ~3 Go de VRAM et
~1 s de latence en moins).

  micro ──WebRTC──▶ VAD (fastrtc ReplyOnPause) ──▶ audio-in natif (conformer)
        ◀──WebRTC── chunks 24 kHz streamés (stage 1 DELTA, TTFA ~180 ms)

Pod GPU :
    python scripts/s2s_webrtc_demo.py --port 7860
    # accès : ssh -L 7860:localhost:7860 → http://localhost:7860

Colab — WebRTC exige un serveur TURN pour traverser le NAT de la VM (le
tunnel gradio.live ne porte que la page, pas le flux média). fastrtc fournit
des credentials TURN Cloudflare GRATUITS via un token Hugging Face :
    import os; os.environ["HF_TOKEN"] = "hf_..."   # ou secret Colab 🔑
    !pip install -q fastrtc
    !python scripts/s2s_webrtc_demo.py --share \\
        --checkpoint /content/lfm25_audio_omni
    # → ouvrir le lien *.gradio.live, autoriser le micro, parler.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2s_demo import REPO, SR_OUT, VllmBackend  # noqa: E402

LOCK = threading.Lock()  # un seul tour à la fois : historique global


def build_stream(backend: VllmBackend, turn: str):
    import gradio as gr
    from fastrtc import AdditionalOutputs, ReplyOnPause, Stream

    def handler(audio: tuple[int, np.ndarray]):
        sr, pcm = audio
        wave = np.asarray(pcm, dtype=np.float32).reshape(-1) / 32_768.0
        with LOCK:
            t0, first, samples = time.time(), None, 0
            try:
                for chunk in backend.reply_stream(audio=(wave, sr)):
                    first = first or (time.time() - t0)
                    samples += chunk.size
                    pcm16 = (np.clip(chunk, -1.0, 1.0) * 32_767).astype(np.int16)
                    yield (SR_OUT, pcm16.reshape(1, -1))
            except ValueError as e:  # ex. « audio trop court » (faux départ VAD)
                print(f"⚠️  tour ignoré : {e}", flush=True)
                return
            txt, m = backend.last_text, backend.last_metrics
            ttfa = f"{first*1000:.0f} ms" if first else "—"
            print(f"🤖 {txt}\n   1er son {ttfa} après fin de parole · "
                  f"{samples/SR_OUT:.1f}s d'audio · gen {m['total_s']:.1f}s", flush=True)
            yield AdditionalOutputs(f"🤖 {txt}\n⏱ 1er son : {ttfa}")

    rtc_conf = server_conf = None
    if turn == "cloudflare":
        from fastrtc import (
            get_cloudflare_turn_credentials,
            get_cloudflare_turn_credentials_async,
        )

        rtc_conf = get_cloudflare_turn_credentials_async  # côté navigateur
        server_conf = get_cloudflare_turn_credentials(ttl=360_000)  # côté VM

    return Stream(
        handler=ReplyOnPause(handler, can_interrupt=False, output_sample_rate=SR_OUT),
        modality="audio",
        mode="send-receive",
        rtc_configuration=rtc_conf,
        server_rtc_configuration=server_conf,
        additional_outputs=[gr.Textbox(label="conversation", lines=6)],
        additional_outputs_handler=lambda old, new: ((old + "\n") if old else "") + new,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="/workspace/models/lfm25_audio_omni")
    ap.add_argument("--deploy-config", type=Path,
                    default=REPO / "configs/vllm_omni_lfm2_audio.yaml")
    ap.add_argument("--no-deploy-config", action="store_true")
    ap.add_argument("--turn", choices=["auto", "cloudflare", "none"], default="auto",
                    help="TURN pour traverser le NAT (Colab) ; auto = cloudflare "
                         "si HF_TOKEN est défini, sinon aucun (réseau local/SSH)")
    ap.add_argument("--share", action="store_true",
                    help="tunnel public gradio.live (requis sur Colab : HTTPS → micro)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    turn = args.turn
    if turn == "auto":
        turn = "cloudflare" if os.environ.get("HF_TOKEN") else "none"
    if args.share and turn == "none":
        print("⚠️  --share sans TURN : sur Colab le flux WebRTC ne passera "
              "probablement pas le NAT — définir HF_TOKEN (TURN Cloudflare gratuit).")

    backend = VllmBackend(
        args.checkpoint,
        deploy_config=None if args.no_deploy_config else args.deploy_config,
    )
    backend.system += " Keep your spoken answers short and conversational."
    # chauffe : JIT Triton + captures CUDA graph avant le 1er utilisateur
    for i in range(2):
        t0 = time.time()
        backend.reply(text="Hello! Please answer briefly.")
        backend.reset()
        print(f"[warmup {i+1}/2] {time.time()-t0:.1f}s", flush=True)

    stream = build_stream(backend, turn)
    print(f"\n▶ démo WebRTC mains-libres prête (TURN: {turn})", flush=True)
    stream.ui.launch(server_port=args.port, share=args.share, quiet=True)


if __name__ == "__main__":
    main()
