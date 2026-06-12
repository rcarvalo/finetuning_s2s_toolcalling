#!/usr/bin/env python3
"""Démo Gradio speech-to-speech (Colab-ready) : micro → LFM2.5-Audio → voix.

Audio-in NATIF vLLM-Omni (un seul engine, pas d'ASR séparé) et réponse
STREAMÉE : la voix commence à jouer ~200 ms après la fin de l'enregistrement
(TTFA mesuré 180 ms, cf. docs/optimization_audit.md), pendant que la suite
se génère (RTF ~0,34 : jamais de trou).

Colab :
    !pip install -q gradio
    !python scripts/s2s_gradio_demo.py --checkpoint /content/lfm25_audio_omni --share
    # → ouvrir le lien public *.gradio.live (HTTPS : le micro marche)

UX : enregistre (bouton micro), arrête → la réponse se joue en streaming.
L'historique de conversation est GLOBAL (démo mono-session ; les tours sont
sérialisés par la queue Gradio) — bouton Reset pour repartir à zéro.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2s_demo import REPO, SR_OUT, VllmBackend  # noqa: E402


def _mic_to_wav(mic: tuple[int, np.ndarray]) -> Path:
    """(sr, pcm du navigateur) → WAV mono float32 temporaire."""
    import soundfile as sf

    sr, data = mic
    if data.dtype != np.float32:
        data = data.astype(np.float32) / 32_768.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, data, sr)
    return Path(f.name)


def build_ui(backend: VllmBackend):
    import gradio as gr

    def on_turn(mic, history):
        """Tour de parole : stream les chunks audio, puis le texte final."""
        history = history or []
        if mic is None:
            yield gr.update(), history, gr.update()
            return
        wav_path = _mic_to_wav(mic)
        t0 = time.time()
        try:
            for chunk in backend.reply_stream(audio_path=wav_path):
                pcm16 = (np.clip(chunk, -1.0, 1.0) * 32_767).astype(np.int16)
                yield (SR_OUT, pcm16), history, gr.update()
        finally:
            wav_path.unlink(missing_ok=True)
        m = backend.last_metrics
        ttfa = f"{m['ttfa_s']*1000:.0f} ms" if m.get("ttfa_s") else "—"
        history = history + [
            {"role": "user", "content": "🎤 (audio)"},
            {"role": "assistant", "content": backend.last_text},
        ]
        status = f"TTFA {ttfa} · total {time.time()-t0:.1f} s"
        yield gr.update(), history, status

    def on_reset():
        backend.reset()
        return [], "historique vidé"

    with gr.Blocks(title="LFM2.5-Audio S2S") as demo:
        gr.Markdown("# 🎙️ LFM2.5-Audio — speech to speech (vLLM-Omni)\n"
                    "Enregistre, arrête : la réponse se joue en streaming.")
        chat = gr.Chatbot(label="Conversation", height=300, type="messages")
        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Parle ici")
            out = gr.Audio(streaming=True, autoplay=True, label="🔊 Réponse")
        status = gr.Markdown("")
        reset = gr.Button("Reset")

        mic.stop_recording(on_turn, inputs=[mic, chat], outputs=[out, chat, status])
        reset.click(on_reset, outputs=[chat, status])

    # 1 tour à la fois : l'historique du backend est global
    demo.queue(default_concurrency_limit=1)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="/workspace/models/lfm25_audio_omni")
    ap.add_argument("--deploy-config", type=Path,
                    default=REPO / "configs/vllm_omni_lfm2_audio.yaml")
    ap.add_argument("--no-deploy-config", action="store_true")
    ap.add_argument("--share", action="store_true",
                    help="tunnel public gradio.live (requis sur Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    backend = VllmBackend(
        args.checkpoint,
        deploy_config=None if args.no_deploy_config else args.deploy_config,
    )
    # chauffe AVANT le 1er utilisateur : JIT Triton + captures CUDA graph
    # (depthformer + buckets) — sinon le 1er tour paie ~1,5 s de plus
    for i in range(2):
        t0 = time.time()
        backend.reply(text="Hello! Please answer briefly.")
        backend.reset()
        print(f"[warmup {i+1}/2] {time.time()-t0:.1f}s", flush=True)

    demo = build_ui(backend)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
