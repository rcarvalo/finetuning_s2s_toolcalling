#!/usr/bin/env python3
"""Démo S2S + tool calling MAINS-LIBRES (WebRTC) — web_search LIVE + fake DB.

Tu parles → le modèle émet le tool call en TEXTE (génération sequential, propre),
l'orchestrateur exécute l'outil (web_search via ddgs, db_query sur une fake DB),
réinjecte le résultat, puis le modèle **parle** la réponse ancrée en INTERLEAVED
(parole S2S basse latence). Agent HYBRIDE. Même UX que ``s2s_webrtc_demo.py``
(FastRTC + VAD + TURN Cloudflare).

L'activité outil s'affiche dans le transcript pendant le round-trip (court) ;
la réponse vocale suit en streaming. « Penser en texte, parler en audio ».

  python scripts/s2s_toolcalling_webrtc_demo.py --share --turn cloudflare \
      --checkpoint LiquidAI/LFM2.5-Audio-1.5B --adapter <dir|HF repo de l'adaptateur Phase B>

Sur Colab : CLOUDFLARE_TURN_KEY_ID/_API_TOKEN comme pour s2s_webrtc_demo.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

# Le décodage audio mimi (streaming) déclenche une capture CUDA-graph inductor
# qui plante (« Cannot copy CPU<->CUDA during CUDA graph capture »). On force
# l'eager (avant tout import torch) : un peu plus lent mais robuste pour la démo.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SR_OUT = 24_000
LOCK = threading.Lock()  # un seul tour à la fois (session globale)


def _resolve_adapter(adapter: str | None) -> str | None:
    """Chemin local tel quel, ou télécharge un repo HF d'adaptateur."""
    if not adapter or os.path.isdir(adapter):
        return adapter
    from huggingface_hub import snapshot_download

    return snapshot_download(adapter)


def _tts(model, proc, text: str) -> "np.ndarray | None":
    """TTS d'une phrase avec le modèle (mode 'Perform TTS') → waveform 24 kHz."""
    import torch
    from liquid_audio import ChatState

    chat = ChatState(proc)
    chat.new_turn("system"); chat.add_text("Perform TTS."); chat.end_turn()
    chat.new_turn("user"); chat.add_text(text); chat.end_turn()
    chat.new_turn("assistant")
    mimi = proc.mimi.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad(), mimi.streaming(1):
        for t in model.generate_interleaved(**chat, max_new_tokens=400, audio_temperature=0.6, audio_top_k=4):
            if t.numel() == 8 and not bool((t == 2048).any()):
                chunks.append(mimi.decode(t[None, :, None])[0].float().cpu().numpy().reshape(-1))
    return np.concatenate(chunks) if chunks else None


def _render_fillers(model, proc, bank, out_dir: Path) -> None:
    """Pré-rend les wavs de filler (une fois) → joués INSTANTANÉMENT pendant le
    round-trip outil pour garder la latence perçue de l'interleaved S2S."""
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    for tool, phrases in bank.phrases.items():
        name = "default" if tool == "_default" else tool
        for i, phrase in enumerate(phrases):
            try:
                wav = _tts(model, proc, phrase)
                if wav is not None and wav.size:
                    sf.write(str(out_dir / f"{name}_{i}.wav"), wav, SR_OUT, subtype="PCM_16")
            except Exception as e:  # noqa: BLE001 — un filler raté n'empêche pas la démo
                print(f"[filler] échec rendu '{phrase}': {e}", flush=True)
    print(f"[filler] wavs pré-rendus dans {out_dir}", flush=True)


def build_agent(checkpoint: str, adapter: str | None):
    import torch  # noqa: F401
    from s2s_demo import LiquidBackend
    from s2s_toolcalling.data.chat_format import TOOLCALLING_EN_SYSTEM_INSTRUCTIONS
    from s2s_toolcalling.orchestrator.agent import AgentConfig, ReceptionAgent
    from s2s_toolcalling.orchestrator.fillers import EN_FILLER_PHRASES, FillerBank
    from s2s_toolcalling.tools.fake_db import FakeDbBackend
    from s2s_toolcalling.tools.toolcalling_en import build_toolcalling_en_registry
    from s2s_toolcalling.tools.web_search import DuckDuckGoBackend

    lb = LiquidBackend(checkpoint, adapter=_resolve_adapter(adapter))  # charge modèle + proc (+ merge LoRA)
    registry = build_toolcalling_en_registry(web_backend=DuckDuckGoBackend(max_results=4), db_backend=FakeDbBackend())
    config = AgentConfig(
        system_instructions=(TOOLCALLING_EN_SYSTEM_INSTRUCTIONS + " Keep spoken answers short."),
        # hybrid=True par défaut : tool call en sequential (texte propre) PUIS
        # réponse en interleaved (parole S2S basse latence).
    )
    # Fillers EN pré-rendus (voix du modèle) → joués pendant le round-trip outil.
    filler_dir = Path("/tmp/fillers_en")
    bank = FillerBank(filler_dir=filler_dir, phrases=dict(EN_FILLER_PHRASES))
    _render_fillers(lb.model, lb.proc, bank, filler_dir)
    return ReceptionAgent(lb.model, lb.proc, registry, config=config, fillers=bank)


def build_stream(agent, turn: str):
    import gradio as gr
    import torch
    from fastrtc import AdditionalOutputs, ReplyOnPause, Stream
    from s2s_toolcalling.orchestrator.events import (
        AudioChunk,
        FillerSpeech,
        ToolCallBegin,
        ToolCallResult,
        TurnComplete,
    )

    session = agent.new_session()  # contexte multi-tours partagé

    def handler(audio: tuple[int, np.ndarray]):
        sr, pcm = audio
        pcm = np.asarray(pcm)
        wave = (pcm.astype(np.float32).reshape(-1) / 32_768.0) if np.issubdtype(pcm.dtype, np.integer) \
            else pcm.astype(np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(wave**2))) if wave.size else 0.0
        print(f"👤 entrée {wave.size/sr:.1f}s @ {sr}Hz · RMS {rms:.3f}", flush=True)
        with LOCK:
            transcript: list[str] = []
            for ev in agent.respond(session, torch.from_numpy(wave).reshape(1, -1), sr):
                if isinstance(ev, ToolCallBegin):
                    line = f"🔧 {ev.name}({ev.arguments})"
                    print(line, flush=True)
                    yield AdditionalOutputs(line)
                elif isinstance(ev, ToolCallResult):
                    yield AdditionalOutputs(f"   ↳ {str(ev.payload)[:120]}")
                elif isinstance(ev, FillerSpeech):
                    yield AdditionalOutputs(f"💬 {ev.phrase}")
                    if ev.wav_path:  # joué pendant le round-trip → masque la latence outil
                        import soundfile as sf

                        fwav, _ = sf.read(ev.wav_path, dtype="float32")
                        pcm16 = (np.clip(fwav.reshape(-1), -1.0, 1.0) * 32_767).astype(np.int16)
                        yield (SR_OUT, pcm16.reshape(1, -1))
                elif isinstance(ev, AudioChunk):
                    pcm16 = (np.clip(ev.samples.numpy().reshape(-1), -1.0, 1.0) * 32_767).astype(np.int16)
                    yield (SR_OUT, pcm16.reshape(1, -1))
                elif isinstance(ev, TurnComplete):
                    transcript.append(ev.text)
                    print(f"🤖 {ev.text}  ({ev.tool_rounds} tool round(s))", flush=True)
                    yield AdditionalOutputs(f"🤖 {ev.text}")

    rtc_conf = server_conf = None
    if turn == "cloudflare":
        from fastrtc import get_cloudflare_turn_credentials

        key_id, key_token = os.environ.get("CLOUDFLARE_TURN_KEY_ID"), os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN")
        if key_id and key_token:
            creds = get_cloudflare_turn_credentials(turn_key_id=key_id, turn_key_api_token=key_token, ttl=86_400)
            rtc_conf = server_conf = creds
            print("[TURN] Cloudflare clés directes", flush=True)

    reply_kwargs: dict = {"can_interrupt": True, "output_sample_rate": SR_OUT}
    try:
        from fastrtc import AlgoOptions, SileroVadOptions

        reply_kwargs["algo_options"] = AlgoOptions(audio_chunk_duration=0.6, started_talking_threshold=0.2,
                                                   speech_threshold=0.1)
        reply_kwargs["model_options"] = SileroVadOptions(threshold=0.5, min_silence_duration_ms=900, speech_pad_ms=300)
    except (ImportError, TypeError) as e:
        print(f"[VAD] défauts ({e})", flush=True)

    return Stream(
        handler=ReplyOnPause(handler, **reply_kwargs),
        modality="audio", mode="send-receive",
        rtc_configuration=rtc_conf, server_rtc_configuration=server_conf,
        additional_outputs=[gr.Textbox(label="conversation + outils", lines=10)],
        additional_outputs_handler=lambda old, new: ((old + "\n") if old else "") + new,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="LiquidAI/LFM2.5-Audio-1.5B")
    ap.add_argument("--adapter", default=None, help="dir local ou repo HF de l'adaptateur (Phase B de préférence)")
    ap.add_argument("--turn", choices=["auto", "cloudflare", "none"], default="auto")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    turn = args.turn
    if turn == "auto":
        turn = "cloudflare" if (os.environ.get("CLOUDFLARE_TURN_KEY_ID") and
                                os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN")) else "none"

    agent = build_agent(args.checkpoint, args.adapter)
    stream = build_stream(agent, turn)
    print(f"\n▶ démo S2S + tool calling prête (TURN: {turn})", flush=True)
    stream.ui.launch(server_port=args.port, share=args.share, quiet=True)


if __name__ == "__main__":
    main()
