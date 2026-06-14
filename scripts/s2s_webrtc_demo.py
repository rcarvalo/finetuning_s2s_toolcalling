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
tunnel gradio.live ne porte que la page, pas le flux média). Sur Colab,
l'endpoint TURN par HF_TOKEN (CLOUDFLARE_FASTRTC_TURN_URL) ne résout souvent
PAS → utiliser des **clés Cloudflare Realtime DIRECTES** (gratuites) :
    1. https://dash.cloudflare.com → Calls → TURN → créer une clé
       → récupérer Turn Token ID + API Token.
    2. Colab :
        import os
        os.environ["CLOUDFLARE_TURN_KEY_ID"] = "..."
        os.environ["CLOUDFLARE_TURN_KEY_API_TOKEN"] = "..."
        os.environ.pop("HF_TOKEN", None)   # sinon il prend le pas (endpoint HF)
        !{sys.executable} -m pip install -q fastrtc
        !{sys.executable} scripts/s2s_webrtc_demo.py --share --turn cloudflare \\
            --checkpoint /content/lfm25_audio_omni
    # logs attendus : « [TURN] Cloudflare clés directes » → ouvrir le lien
    # *.gradio.live, autoriser le micro, parler.
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


def build_stream(backend: VllmBackend, turn: str, *,
                 silence_ms: int = 700, speech_pad_ms: int = 300,
                 vad_threshold: float = 0.5, can_interrupt: bool = True):
    import gradio as gr
    from fastrtc import AdditionalOutputs, ReplyOnPause, Stream

    def handler(audio: tuple[int, np.ndarray]):
        sr, pcm = audio
        pcm = np.asarray(pcm)
        # fastrtc donne du int16 (souvent) ou du float ; normalise sans écraser
        if np.issubdtype(pcm.dtype, np.integer):
            wave = pcm.astype(np.float32).reshape(-1) / 32_768.0
        else:
            wave = pcm.astype(np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(wave**2))) if wave.size else 0.0
        # diagnostic d'ENTRÉE : durée trop courte = VAD qui coupe ; RMS trop bas
        # = micro/gain (le modèle « n'entend » rien → réponses génériques)
        print(f"👤 entrée {wave.size/sr:.1f}s @ {sr}Hz · niveau RMS {rms:.3f}"
              + ("  ⚠️ très faible (gain micro ?)" if rms < 0.01 else ""), flush=True)
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

        key_id = os.environ.get("CLOUDFLARE_TURN_KEY_ID")
        key_token = os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN")
        if key_id and key_token:
            # Clés Cloudflare Realtime DIRECTES → rtc.live.cloudflare.com
            # (contourne CLOUDFLARE_FASTRTC_TURN_URL, l'endpoint HF que la VM
            # Colab ne résout pas). On récupère les ICE servers une fois (ttl
            # 24 h = max Cloudflare) et on sert le MÊME dict au navigateur et à
            # la VM — pas de dépendance à la variante async côté client.
            creds = get_cloudflare_turn_credentials(
                turn_key_id=key_id, turn_key_api_token=key_token, ttl=86_400
            )
            rtc_conf = creds  # navigateur
            server_conf = creds  # VM
            print("[TURN] Cloudflare clés directes (rtc.live.cloudflare.com)", flush=True)
        else:
            # repli : endpoint HF via HF_TOKEN (peut ne pas résoudre sur Colab)
            rtc_conf = get_cloudflare_turn_credentials_async
            server_conf = get_cloudflare_turn_credentials(ttl=360_000)
            print("[TURN] endpoint HF (HF_TOKEN) — peut échouer sur Colab", flush=True)

    # VAD assoupli : le défaut fastrtc (min_silence_duration_ms=100) coupe au
    # moindre micro-silence → le modèle ne reçoit qu'un fragment. On laisse une
    # vraie pause (silence_ms) avant de clore le tour, + du padding pour ne pas
    # rogner les bords de la parole. Construit défensivement (les champs varient
    # selon la version de fastrtc).
    reply_kwargs: dict = {"can_interrupt": can_interrupt, "output_sample_rate": SR_OUT}
    try:
        from fastrtc import AlgoOptions, SileroVadOptions

        reply_kwargs["algo_options"] = AlgoOptions(
            audio_chunk_duration=0.6,
            started_talking_threshold=0.2,
            speech_threshold=0.1,
        )
        reply_kwargs["model_options"] = SileroVadOptions(
            threshold=vad_threshold,
            min_silence_duration_ms=silence_ms,  # ← laisse finir la phrase
            speech_pad_ms=speech_pad_ms,         # ← ne rogne pas les bords
        )
        print(f"[VAD] silence_fin_de_tour={silence_ms}ms · pad={speech_pad_ms}ms · "
              f"seuil={vad_threshold} · barge-in={can_interrupt}", flush=True)
    except (ImportError, TypeError) as e:
        print(f"[VAD] options par défaut (réglage fin indisponible : {e})", flush=True)

    return Stream(
        handler=ReplyOnPause(handler, **reply_kwargs),
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
    # réglage du turn-taking (anti « elle me coupe »)
    ap.add_argument("--vad-silence-ms", type=int, default=700,
                    help="silence (ms) avant de clore ton tour ; ↑ si elle te coupe")
    ap.add_argument("--speech-pad-ms", type=int, default=300,
                    help="padding gardé autour de la parole (évite de rogner les bords)")
    ap.add_argument("--vad-threshold", type=float, default=0.5,
                    help="seuil de détection de parole silero (↑ = moins sensible au bruit)")
    ap.add_argument("--no-interrupt", action="store_true",
                    help="désactive le barge-in (par défaut tu peux couper l'assistant)")
    args = ap.parse_args()

    turn = args.turn
    if turn == "auto":
        has_cf_keys = bool(os.environ.get("CLOUDFLARE_TURN_KEY_ID")
                           and os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN"))
        turn = "cloudflare" if (has_cf_keys or os.environ.get("HF_TOKEN")) else "none"
    if args.share and turn == "none":
        print("⚠️  --share sans TURN : sur Colab le flux WebRTC ne passera "
              "probablement pas le NAT — définir CLOUDFLARE_TURN_KEY_ID + "
              "CLOUDFLARE_TURN_KEY_API_TOKEN (clés Realtime gratuites).")

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

    stream = build_stream(
        backend, turn,
        silence_ms=args.vad_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        vad_threshold=args.vad_threshold,
        can_interrupt=not args.no_interrupt,
    )
    print(f"\n▶ démo WebRTC mains-libres prête (TURN: {turn})", flush=True)
    stream.ui.launch(server_port=args.port, share=args.share, quiet=True)


if __name__ == "__main__":
    main()
