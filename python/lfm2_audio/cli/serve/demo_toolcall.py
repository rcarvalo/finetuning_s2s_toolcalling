#!/usr/bin/env python3
"""Démo S2S + tool calling sur vLLM-Omni (web_search + db_query parlés).

    lfm2-toolcalling-demo --checkpoint exports/tc_en --adapter Rcarvalo/lfm25-tc-en-s2s-adapter
    lfm2-toolcalling-demo --checkpoint exports/tc_en_omni --share

Le checkpoint est résolu automatiquement : chemin local, repo Hugging Face, ou
base + adaptateur LoRA (fusion et conversion faites une fois, puis mises en
cache). ``TAVILY_API_KEY`` donne un web_search propre, sinon repli DuckDuckGo.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Any

# TORCHDYNAMO_DISABLE was set here historically (all-eager era). It is fatal
# now: the deploy config compiles stage 0, and disabling dynamo kills that
# stage at boot with 'aot_compile is not supported' (verified on L4, 08-23).
import gradio as gr
import numpy as np
import soundfile as sf
from fastrtc import (
    AdditionalOutputs,
    AlgoOptions,
    ReplyOnPause,
    SileroVadOptions,
    Stream,
    get_cloudflare_turn_credentials,
)

from lfm2_audio.core.env import preload_cuda13
from lfm2_audio.core.errors import BackendUnavailableError
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.inference_config import EngineConfig
from lfm2_audio.orchestrator.events import (
    AudioChunk,
    FillerSpeech,
    ToolCallBegin,
    ToolCallResult,
    TurnComplete,
)
from lfm2_audio.orchestrator.fillers import EN_FILLER_PHRASES, FillerBank
from lfm2_audio.orchestrator.vllm_tool_agent import VllmToolAgent
from lfm2_audio.serving.backends.vllm_omni import VllmOmniBackend
from lfm2_audio.tools.fake_db import FakeDbBackend
from lfm2_audio.tools.toolcalling_en import build_toolcalling_en_registry
from lfm2_audio.tools.web_search.duckduckgo import DuckDuckGoBackend
from lfm2_audio.tools.web_search.tavily import TavilyBackend

SR_OUT = 24_000
# Sous ce RMS : écho (sortie modèle reprise au micro) ou silence → on ignore.
MIN_INPUT_RMS = float(os.environ.get("MIN_INPUT_RMS", "0.03"))
LOCK = threading.Lock()


# ───────────────────────── adaptateur backend vLLM-Omni ──────────────────────


# ──────────────────────────────── construction ───────────────────────────────


def build_agent(
    checkpoint: str,
    adapter: str | None = None,
    *,
    no_deploy_config: bool = False,
    filler_dir: str | None = None,
) -> VllmToolAgent:
    """Assemble modèle + registre d'outils + fillers en un agent prêt à répondre."""

    preload_cuda13()  # AVANT tout import de vllm
    model = VllmOmniBackend.from_pretrained(
        checkpoint,
        backend="vllm",
        adapter=adapter,
        engine=EngineConfig(deploy_config=None) if no_deploy_config else EngineConfig(),
    )
    # La fabrique retourne l'ABC : l'agent a besoin de `stream_turns`, que seul
    # le backend vLLM expose. Narrowing explicite plutôt qu'un cast silencieux.
    if not isinstance(model, VllmOmniBackend):
        message = f"la démo tool-calling exige le backend vLLM-Omni, obtenu {type(model).__name__}"
        raise BackendUnavailableError(message)

    # Tavily : 2 résultats suffisent (le handler borne à 2) et c'est plus rapide.
    web = TavilyBackend(max_results=2) if os.environ.get("TAVILY_API_KEY") else DuckDuckGoBackend(max_results=4)
    print(f"[web] {'Tavily' if os.environ.get('TAVILY_API_KEY') else 'DuckDuckGo (repli)'}", flush=True)
    registry = build_toolcalling_en_registry(web_backend=web, db_backend=FakeDbBackend())
    # Fillers : wavs PRÉ-RENDUS si --filler-dir est fourni (voix Aiden — celle
    # que le modèle a apprise, donc transition invisible), texte seul sinon.
    # Ne JAMAIS les rendre via « Perform TTS. » sur le modèle tool-calling :
    # c'est hors-distribution → il émet un token audio en step TEXT → la machine
    # de modalité lève et TUE l'engine.
    bank = FillerBank(
        filler_dir=Path(filler_dir) if filler_dir else None,
        phrases=dict(EN_FILLER_PHRASES),
    )
    return VllmToolAgent(model, registry, fillers=bank)


def build_stream(agent: VllmToolAgent, turn: str) -> Any:

    def handler(audio: tuple[int, np.ndarray]) -> Any:

        sample_rate, pcm = audio
        wave = Waveform.from_pcm16(np.asarray(pcm), sample_rate)
        print(
            f"👤 entrée {wave.duration_s:.1f}s @ {wave.sample_rate}Hz · RMS {wave.rms:.3f}",
            flush=True,
        )
        if wave.rms < MIN_INPUT_RMS:
            print(f"   (ignoré : RMS {wave.rms:.3f} < {MIN_INPUT_RMS} — écho/silence)", flush=True)
            return
        # Chronométrage par étape : c'est la donnée que l'écoute seule ne donne
        # pas — où partent les secondes d'un tour (décision, outil, reprise).
        t_start = time.monotonic()
        t_mark = t_start
        first_sound_s: float | None = None
        with LOCK:
            # NB : pas de sonde « transcription » — le modèle Phase B ne transcrit
            # plus (le fine-tuning tool-calling a écrasé l'ASR : il RÉPOND au lieu
            # de transcrire). Le vrai « ce qu'il a compris » = la requête du tool
            # call (ligne 🔧), qui est fidèle (« weather in Paris », « in Celsius »…).
            for ev in agent.respond(wave):
                if isinstance(ev, ToolCallBegin):
                    decision_s = time.monotonic() - t_mark
                    line = f"🔧 {ev.name}({ev.arguments})  ⏱ décision {decision_s:.2f}s"
                    print(line, flush=True)
                    yield AdditionalOutputs(line)
                elif isinstance(ev, ToolCallResult):
                    t_mark = time.monotonic()  # la reprise se mesure depuis le résultat
                    print(f"   ↳ outil ok={ev.ok} ({ev.elapsed_ms:.0f}ms) → {str(ev.payload)[:300]}", flush=True)
                    yield AdditionalOutputs(f"   ↳ ⏱ outil {ev.elapsed_ms:.0f}ms · {str(ev.payload)[:140]}")
                elif isinstance(ev, FillerSpeech):
                    yield AdditionalOutputs(f"💬 {ev.phrase}")
                    if ev.wav_path:  # joué pendant le round-trip → masque le TTFA
                        fwav, _ = sf.read(ev.wav_path, dtype="float32")
                        pcm16 = (np.clip(fwav.reshape(-1), -1.0, 1.0) * 32_767).astype(np.int16)
                        yield (SR_OUT, pcm16.reshape(1, -1))
                elif isinstance(ev, AudioChunk):
                    if first_sound_s is None:
                        first_sound_s = time.monotonic() - t_mark
                    # vLLM RTF<1 → on stream EN DIRECT (pas d'underrun, contrairement
                    # à liquid-audio) : c'est tout l'intérêt du port.
                    w = np.asarray(ev.samples, dtype=np.float32).reshape(-1)
                    pcm16 = (np.clip(w, -1.0, 1.0) * 32_767).astype(np.int16)
                    yield (SR_OUT, pcm16.reshape(1, -1))
                elif isinstance(ev, TurnComplete):
                    total_s = time.monotonic() - t_start
                    ttfa = f"{first_sound_s:.2f}s" if first_sound_s is not None else "—"
                    print(
                        f"🤖 {ev.text}  ({ev.tool_rounds} round(s), 1er son {ttfa}, total {total_s:.1f}s)",
                        flush=True,
                    )
                    yield AdditionalOutputs(f"🤖 {ev.text}\n⏱ 1er son après outil {ttfa} · tour complet {total_s:.1f}s")

    rtc_conf = server_conf = None
    if turn == "cloudflare":
        key_id, key_token = os.environ.get("CLOUDFLARE_TURN_KEY_ID"), os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN")
        if key_id and key_token:
            creds = get_cloudflare_turn_credentials(turn_key_id=key_id, turn_key_api_token=key_token, ttl=86_400)
            rtc_conf = server_conf = creds
            print("[TURN] Cloudflare clés directes", flush=True)

    reply_kwargs: dict = {"can_interrupt": False, "output_sample_rate": SR_OUT}
    try:
        reply_kwargs["algo_options"] = AlgoOptions(
            audio_chunk_duration=0.6, started_talking_threshold=0.2, speech_threshold=0.1
        )
        reply_kwargs["model_options"] = SileroVadOptions(threshold=0.5, min_silence_duration_ms=900, speech_pad_ms=300)
    except (ImportError, TypeError) as e:
        print(f"[VAD] défauts ({e})", flush=True)

    return Stream(
        handler=ReplyOnPause(handler, **reply_kwargs),
        modality="audio",
        mode="send-receive",
        rtc_configuration=rtc_conf,
        server_rtc_configuration=server_conf,
        additional_outputs=[gr.Textbox(label="conversation + outils", lines=10)],
        additional_outputs_handler=lambda old, new: ((old + "\n") if old else "") + new,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="répertoire local, repo Hugging Face, ou adaptateur (conversion automatique)",
    )
    ap.add_argument("--adapter", default=None, help="adaptateur LoRA à fusionner dans la base")
    ap.add_argument(
        "--no-deploy-config",
        action="store_true",
        help="kwargs legacy (tout eager) au lieu du YAML par stage — TTFA dégradé",
    )
    ap.add_argument(
        "--filler-dir",
        default=None,
        help="wavs d'attente pré-rendus (voix du modèle) joués pendant l'exécution de l'outil",
    )
    ap.add_argument("--turn", choices=["auto", "cloudflare", "none"], default="auto")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    turn = args.turn
    if turn == "auto":
        turn = (
            "cloudflare"
            if (os.environ.get("CLOUDFLARE_TURN_KEY_ID") and os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN"))
            else "none"
        )

    agent = build_agent(
        args.checkpoint,
        args.adapter,
        no_deploy_config=args.no_deploy_config,
        filler_dir=args.filler_dir,
    )
    stream = build_stream(agent, turn)
    print(f"\n▶ démo S2S + tool calling vLLM-Omni prête (TURN: {turn})", flush=True)
    stream.ui.launch(server_port=args.port, share=args.share, quiet=True)


if __name__ == "__main__":
    main()
