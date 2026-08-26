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

from lfm2_audio.cli.serve.turn_io import LOCK, as_pcm16
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
from lfm2_audio.orchestrator.playback import detach
from lfm2_audio.tools.fake_db import FakeDbBackend
from lfm2_audio.tools.toolcalling_en import build_toolcalling_en_registry
from lfm2_audio.tools.web_search.duckduckgo import DuckDuckGoBackend

# ───────────────────────── adaptateur backend vLLM-Omni ──────────────────────


# ──────────────────────────────── construction ───────────────────────────────


def _build_registry() -> Any:
    """Outils réels : Tavily si sa clé est là (plus rapide), DuckDuckGo sinon.

    Le module Tavily n'est résolu que dans sa branche : il tire le SDK
    ``tavily``, que le repli DuckDuckGo n'a aucune raison d'exiger — même
    motif que les imports vLLM de ``build_agent``.
    """
    web: Any
    if os.environ.get("TAVILY_API_KEY"):
        from lfm2_audio.tools.web_search.tavily import TavilyBackend

        web = TavilyBackend(max_results=2)
        print("[web] Tavily", flush=True)
    else:
        web = DuckDuckGoBackend(max_results=4)
        print("[web] DuckDuckGo (repli)", flush=True)
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


def stream_turn(agent: Any, wave: Waveform) -> Iterator[tuple[Any, Any, str]]:
    """Play one turn as (filler, reply, trace), each yielded the moment it exists.

    Two players rather than one streamed player. Gradio's streaming audio output
    turns every yield into a complete WAV file (its API reports the output type
    as ``filepath``), so chaining spans chained WAV headers: the reply chopped,
    and the component never recovered for the next turn — turn 2 stayed silent.

    The filler leaves as soon as the tool call is emitted, covering the search,
    and the reply leaves complete. Nothing carries stream state between turns,
    which is what makes every turn behave like the first.
    """
    lines: list[str] = []
    chunks: list[np.ndarray] = []
    start = time.monotonic()
    mark = start
    first_sound: float | None = None

    def turn() -> Iterator[Any]:
        with LOCK:
            yield from agent.respond(wave)

    for event in detach(turn):
        filler_out: Any = gr.skip()
        reply_out: Any = gr.skip()
        if isinstance(event, ToolCallBegin):
            lines.append(f"🔧 {event.name}({event.arguments})  ⏱ décision {time.monotonic() - mark:.2f}s")
        elif isinstance(event, ToolCallResult):
            mark = time.monotonic()
            lines.append(f"   ↳ ⏱ outil {event.elapsed_ms:.0f}ms · {str(event.payload)[:180]}")
        elif isinstance(event, FillerSpeech):
            lines.append(f"💬 {event.phrase}")
            if event.wav_path:
                filler, _ = sf.read(event.wav_path, dtype="float32")
                filler_out = as_pcm16(np.asarray(filler, dtype=np.float32).reshape(-1))
        elif isinstance(event, AudioChunk):
            if first_sound is None:
                first_sound = time.monotonic() - mark
            chunks.append(np.asarray(event.samples, dtype=np.float32).reshape(-1))
            continue  # rien à montrer tant que la réponse n'est pas entière
        elif isinstance(event, TurnComplete):
            ttfa = f"{first_sound:.2f}s" if first_sound is not None else "—"
            lines.append(f"🤖 {event.text}")
            lines.append(f"⏱ 1er son {ttfa} · tour complet {time.monotonic() - start:.1f}s")
            if chunks:
                reply_out = as_pcm16(np.concatenate(chunks))
        yield filler_out, reply_out, "\n".join(lines)


def build_simple_ui(agent: Any) -> Any:
    """Interface enregistrer → envoyer, sans WebRTC ni TURN."""

    def on_submit(recording: tuple[int, np.ndarray] | None, history: str) -> Iterator[tuple[Any, Any, str]]:
        if recording is None:
            yield gr.skip(), gr.skip(), history
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
        for filler_out, reply_out, trace in stream_turn(agent, wave):
            yield filler_out, reply_out, prefix + trace
        print(trace, flush=True)

    with gr.Blocks(title="LFM2.5-Audio · tool calling") as ui:
        gr.Markdown("### Parlez, puis envoyez — les outils sont réels (web + base de démo).")
        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="numpy", label="votre question")
            # Deux lecteurs : l'attente part pendant la recherche, la réponse
            # arrive entière. Aucun état de flux ne survit d'un tour à l'autre.
            filler = gr.Audio(label="pendant la recherche", autoplay=True)
            reply = gr.Audio(label="réponse", autoplay=True)
        trace = gr.Textbox(label="conversation + outils + latences", lines=14)
        outputs = [filler, reply, trace]
        mic.stop_recording(on_submit, inputs=[mic, trace], outputs=outputs)
        gr.Button("Envoyer").click(on_submit, inputs=[mic, trace], outputs=outputs)
    return ui


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
    ap.add_argument("--turn", choices=["auto", "hf", "cloudflare", "none"], default="auto")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    turn = args.turn
    if turn == "auto":
        # HF d'abord : fastrtc sert un relais Cloudflare gratuit contre un token
        # Hugging Face, qu'on a déjà. Sans lui, WebRTC n'a aucun relais joignable
        # (turn.fastrtc.org n'existe plus) et la connexion échoue.
        if os.environ.get("HF_TOKEN"):
            turn = "hf"
        elif os.environ.get("CLOUDFLARE_TURN_KEY_ID") and os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN"):
            turn = "cloudflare"
        else:
            turn = "none"

    agent = build_agent(
        args.checkpoint,
        args.adapter,
        no_deploy_config=args.no_deploy_config,
        filler_dir=args.filler_dir,
        backend=args.backend,
    )
    if args.ui == "simple":
        ui = build_simple_ui(agent)
        mode = "simple (sans WebRTC)"
    else:
        # Résolu ici seulement : ce module tire `fastrtc`, que le chemin simple
        # n'a pas à exiger (cf. le découpage dans `webrtc_ui`).
        from lfm2_audio.cli.serve.webrtc_ui import build_stream

        ui = build_stream(agent, turn).ui
        mode = f"WebRTC (TURN: {turn})"
    print(f"\n▶ démo S2S + tool calling prête — backend {args.backend}, UI {mode}", flush=True)
    # quiet supprime l'affichage de l'URL publique — inutilisable avec --share,
    # dont c'est justement le seul livrable.
    ui.launch(server_port=args.port, share=args.share, quiet=not args.share)


if __name__ == "__main__":
    main()
