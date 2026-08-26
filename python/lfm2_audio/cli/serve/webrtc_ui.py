"""Hands-free voice UI over WebRTC, and the only place that needs ``fastrtc``.

Split out of the demo module so ``--ui simple`` runs on a machine without the
WebRTC stack installed: the browser-tunnel path shares nothing with it but the
constants in :mod:`lfm2_audio.cli.serve.turn_io`.

This path needs a reachable TURN relay — ``turn.fastrtc.org`` no longer exists
and Cloudflare wants a key — which is why the simple UI is the default.
"""

from __future__ import annotations

import os
import time
from typing import Any

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
    get_cloudflare_turn_credentials_async,
)

from lfm2_audio.cli.serve.turn_io import LOCK, MIN_INPUT_RMS, SR_OUT
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.orchestrator.events import (
    AudioChunk,
    FillerSpeech,
    ToolCallBegin,
    ToolCallResult,
    TurnComplete,
)


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
                    # WebRTC absorbe des blocs courts (transport temps réel, pas
                    # un tunnel HTTP) : ici on stream frame par frame sans le
                    # tampon que l'UI simple doit poser.
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

    rtc_conf: Any = None
    server_conf: Any = None
    if turn == "hf":
        # fastrtc sert un relais Cloudflare gratuit contre un token Hugging Face
        # (lu dans HF_TOKEN). Le client reçoit la fonction async — fastrtc la
        # rappelle par session, les identifiants étant à durée de vie courte.
        rtc_conf = get_cloudflare_turn_credentials_async
        server_conf = get_cloudflare_turn_credentials(ttl=86_400)
        print("[TURN] Cloudflare via token Hugging Face", flush=True)
    elif turn == "cloudflare":
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
