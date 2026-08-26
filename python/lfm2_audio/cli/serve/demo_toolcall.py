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
from collections.abc import Iterator
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

from lfm2_audio.core import chat_format
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


def _build_registry() -> Any:
    """Outils réels : Tavily si sa clé est là (plus rapide), DuckDuckGo sinon."""
    web = TavilyBackend(max_results=2) if os.environ.get("TAVILY_API_KEY") else DuckDuckGoBackend(max_results=4)
    print(f"[web] {'Tavily' if os.environ.get('TAVILY_API_KEY') else 'DuckDuckGo (repli)'}", flush=True)
    return build_toolcalling_en_registry(web_backend=web, db_backend=FakeDbBackend())


def _build_fillers(filler_dir: str | None) -> FillerBank:
    """Wavs pré-rendus si fournis (voix du modèle), texte seul sinon.

    Ne JAMAIS les rendre via le modèle tool-calling : c'est hors-distribution,
    il émet un token audio en step TEXT et la machine de modalité tue l'engine.
    """
    return FillerBank(filler_dir=Path(filler_dir) if filler_dir else None, phrases=dict(EN_FILLER_PHRASES))


def build_agent(
    checkpoint: str,
    adapter: str | None = None,
    *,
    no_deploy_config: bool = False,
    filler_dir: str | None = None,
    backend: str = "vllm",
) -> Any:
    """Assemble modèle + registre d'outils + fillers en un agent prêt à répondre."""

    if backend == "liquid":
        # Chemin sans vLLM : aucune dépendance diffusers/peft/torchao, et il
        # porte le correctif deux-passes (réponse parlée quand aucun outil
        # n'est appelé). TTFA plus élevé qu'en vLLM — les fillers le masquent.
        from lfm2_audio.orchestrator.agent import AgentConfig, ReceptionAgent
        from lfm2_audio.orchestrator.liquid_tool_agent import LiquidToolAgent
        from lfm2_audio.serving.backends.liquid import LiquidAudioBackend
        from lfm2_audio.serving.model import LFM2Audio

        served = LFM2Audio.from_pretrained(checkpoint, backend="liquid", adapter=adapter)
        # Même rétrécissement explicite que la branche vLLM : l'agent veut le
        # modèle brut, que seul le backend concret expose (la fabrique rend l'ABC).
        if not isinstance(served, LiquidAudioBackend):
            message = f"--backend liquid exige LiquidAudioBackend, obtenu {type(served).__name__}"
            raise BackendUnavailableError(message)
        reception = ReceptionAgent(
            served._model,
            served._processor,
            _build_registry(),
            config=AgentConfig(max_new_tokens=512, system_instructions=chat_format.TOOLCALLING_EN_SYSTEM_INSTRUCTIONS),
            fillers=_build_fillers(filler_dir),
        )
        return LiquidToolAgent(reception)

    # Imports vLLM DIFFÉRÉS : `core.env` importe vllm_omni au niveau module, et
    # `--backend liquid` doit tourner sur une machine qui n'a ni vLLM ni
    # diffusers. Les charger en tête rendrait le chemin léger dépendant du lourd.
    from lfm2_audio.core.env import preload_cuda13
    from lfm2_audio.orchestrator.vllm_tool_agent import VllmToolAgent
    from lfm2_audio.serving.backends.vllm_omni import VllmOmniBackend

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

    return VllmToolAgent(model, _build_registry(), fillers=_build_fillers(filler_dir))


def run_turn(agent: Any, wave: Waveform) -> tuple[np.ndarray, str]:
    """Un tour complet, rendu en (audio concaténé, trace lisible).

    Chemin SANS WebRTC : l'audio transite par le tunnel HTTPS de Gradio, donc
    aucun relais TURN n'est nécessaire — `turn.fastrtc.org` n'existe plus et
    Cloudflare exige une clé. C'est le mode qui marche partout, au prix du
    mains-libres (on enregistre, puis on envoie).
    """
    lines: list[str] = []
    chunks: list[np.ndarray] = []
    start = time.monotonic()
    mark = start
    first_sound: float | None = None

    with LOCK:
        for event in agent.respond(wave):
            if isinstance(event, ToolCallBegin):
                lines.append(f"🔧 {event.name}({event.arguments})  ⏱ décision {time.monotonic() - mark:.2f}s")
            elif isinstance(event, ToolCallResult):
                mark = time.monotonic()
                lines.append(f"   ↳ ⏱ outil {event.elapsed_ms:.0f}ms · {str(event.payload)[:180]}")
            elif isinstance(event, FillerSpeech):
                lines.append(f"💬 {event.phrase}")
                if event.wav_path:
                    filler, _ = sf.read(event.wav_path, dtype="float32")
                    chunks.append(np.asarray(filler, dtype=np.float32).reshape(-1))
            elif isinstance(event, AudioChunk):
                if first_sound is None:
                    first_sound = time.monotonic() - mark
                chunks.append(np.asarray(event.samples, dtype=np.float32).reshape(-1))
            elif isinstance(event, TurnComplete):
                ttfa = f"{first_sound:.2f}s" if first_sound is not None else "—"
                lines.append(f"🤖 {event.text}")
                lines.append(f"⏱ 1er son après outil {ttfa} · tour complet {time.monotonic() - start:.1f}s")

    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    return audio, "\n".join(lines)


def stream_turn(agent: Any, wave: Waveform) -> Iterator[tuple[Any, str]]:
    """Un tour, rendu AU FIL DE L'EAU : (chunk audio, trace) à chaque événement.

    Attendre la réponse complète coûtait ~7 s perçues alors que le travail utile
    tient en ~2 s (décision + outil + 1er son) : le reste, c'est la synthèse de
    l'audio, qu'on peut écouter pendant qu'elle se fait. Gradio joue les chunks
    à mesure, donc la latence PERÇUE tombe au premier son.
    """
    lines: list[str] = []
    start = time.monotonic()
    mark = start
    first_sound: float | None = None

    with LOCK:
        for event in agent.respond(wave):
            if isinstance(event, ToolCallBegin):
                lines.append(f"🔧 {event.name}({event.arguments})  ⏱ décision {time.monotonic() - mark:.2f}s")
                yield None, "\n".join(lines)
            elif isinstance(event, ToolCallResult):
                mark = time.monotonic()
                lines.append(f"   ↳ ⏱ outil {event.elapsed_ms:.0f}ms · {str(event.payload)[:180]}")
                yield None, "\n".join(lines)
            elif isinstance(event, FillerSpeech):
                lines.append(f"💬 {event.phrase}")
                if event.wav_path:
                    filler, _ = sf.read(event.wav_path, dtype="float32")
                    yield _as_pcm16(np.asarray(filler, dtype=np.float32).reshape(-1)), "\n".join(lines)
                else:
                    yield None, "\n".join(lines)
            elif isinstance(event, AudioChunk):
                if first_sound is None:
                    first_sound = time.monotonic() - mark
                yield _as_pcm16(np.asarray(event.samples, dtype=np.float32).reshape(-1)), "\n".join(lines)
            elif isinstance(event, TurnComplete):
                ttfa = f"{first_sound:.2f}s" if first_sound is not None else "—"
                lines.append(f"🤖 {event.text}")
                lines.append(f"⏱ 1er son {ttfa} · tour complet {time.monotonic() - start:.1f}s")
                yield None, "\n".join(lines)


def _as_pcm16(samples: np.ndarray) -> tuple[int, np.ndarray]:
    return SR_OUT, (np.clip(samples, -1.0, 1.0) * 32_767).astype(np.int16)


def build_simple_ui(agent: Any) -> Any:
    """Interface enregistrer → envoyer, sans WebRTC ni TURN, audio streamé."""

    def on_submit(recording: tuple[int, np.ndarray] | None, history: str) -> Iterator[tuple[Any, str]]:
        if recording is None:
            yield None, history
            return
        sample_rate, pcm = recording
        # `for_encoder` AVANT tout : le micro du navigateur donne du 44,1 kHz et
        # l'encodeur mel est calibré 16 kHz. Le mélange ne lève aucune erreur —
        # il dégrade silencieusement ce que le modèle entend, donc ses décisions.
        wave = Waveform.from_pcm16(np.asarray(pcm), sample_rate).for_encoder()
        print(
            f"👤 entrée {wave.duration_s:.1f}s @ {wave.sample_rate}Hz (micro {sample_rate}Hz) · RMS {wave.rms:.3f}",
            flush=True,
        )
        prefix = (history + "\n\n") if history else ""
        trace = ""
        for chunk, trace in stream_turn(agent, wave):
            yield chunk, prefix + trace
        print(trace, flush=True)

    with gr.Blocks(title="LFM2.5-Audio · tool calling") as ui:
        gr.Markdown("### Parlez, puis envoyez — les outils sont réels (web + base de démo).")
        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="numpy", label="votre question")
            reply = gr.Audio(label="réponse", autoplay=True, streaming=True)
        trace = gr.Textbox(label="conversation + outils + latences", lines=14)
        mic.stop_recording(on_submit, inputs=[mic, trace], outputs=[reply, trace])
        gr.Button("Envoyer").click(on_submit, inputs=[mic, trace], outputs=[reply, trace])
    return ui


def build_stream(agent: Any, turn: str) -> Any:
    """UI voix mains-libres (WebRTC). Exige un relais TURN hors réseau local."""

    def handler(audio: tuple[int, np.ndarray]) -> Any:

        sample_rate, pcm = audio
        # Même règle que l'UI simple : rééchantillonner AVANT l'encodeur.
        wave = Waveform.from_pcm16(np.asarray(pcm), sample_rate).for_encoder()
        print(
            f"👤 entrée {wave.duration_s:.1f}s @ {wave.sample_rate}Hz (micro {sample_rate}Hz) · RMS {wave.rms:.3f}",
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
    ap.add_argument("--backend", choices=["vllm", "liquid"], default="vllm", help="liquid : sans vLLM ni diffusers")
    ap.add_argument(
        "--ui",
        choices=["webrtc", "simple"],
        default="webrtc",
        help="simple : enregistrer→envoyer, sans WebRTC — donc sans relais TURN "
        "(turn.fastrtc.org n'existe plus ; Cloudflare exige une clé)",
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
        backend=args.backend,
    )
    ui = build_simple_ui(agent) if args.ui == "simple" else build_stream(agent, turn).ui
    mode = "simple (sans WebRTC)" if args.ui == "simple" else f"WebRTC (TURN: {turn})"
    print(f"\n▶ démo S2S + tool calling prête — backend {args.backend}, UI {mode}", flush=True)
    # quiet supprime l'affichage de l'URL publique — inutilisable avec --share,
    # dont c'est justement le seul livrable.
    ui.launch(server_port=args.port, share=args.share, quiet=not args.share)


if __name__ == "__main__":
    main()
