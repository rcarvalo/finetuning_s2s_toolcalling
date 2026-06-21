#!/usr/bin/env python3
"""Démo S2S + tool calling sur **vLLM-Omni** (basse latence, RTF<1).

Même boucle outil que la démo liquid (web_search + db_query parlés), mais sur le
moteur vLLM-Omni → streaming temps réel propre (comme la démo S2S de base), pas
l'underrun « voix électrique » de liquid-audio.

PRÉREQUIS — checkpoint Phase B CONVERTI au format omni (le LoRA est pour
liquid-audio ; vLLM charge un checkpoint mergé+converti) :

  # 1) merge LoRA + export checkpoint complet (layout liquid-audio + ratio interleaved)
  python -m s2s_toolcalling.training.export_checkpoint \
      --base LiquidAI/LFM2.5-Audio-1.5B --adapter Rcarvalo/lfm25-tc-en-s2s-adapter \
      --output exports/tc_en --mode full \
      --interleaved-text-tokens 6 --interleaved-audio-tokens 12

  # 2) convert config.json → architecture/pipeline omni
  python -m vllm_omni_lfm2_audio.convert_checkpoint \
      --checkpoint exports/tc_en --output exports/tc_en_omni

  # 3) démo
  python scripts/s2s_toolcalling_vllm_demo.py --checkpoint exports/tc_en_omni --share

Tavily (TAVILY_API_KEY) pour un web_search propre, sinon repli DuckDuckGo.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

SR_OUT = 24_000
# Sous ce RMS : écho (sortie modèle reprise au micro) ou silence → on ignore.
MIN_INPUT_RMS = float(os.environ.get("MIN_INPUT_RMS", "0.03"))
LOCK = threading.Lock()


# ───────────────────────── adaptateur backend vLLM-Omni ──────────────────────


def _build_tool_backend(checkpoint: str, deploy_config: Path | None):
    """``VllmToolBackend`` : sous-classe de ``VllmBackend`` (réutilise le
    chargement Omni) + ``stream(turns, stop_token_ids)`` attendu par l'agent.

    Défini en fabrique pour que l'import lourd de ``s2s_demo`` (et donc vLLM) ne
    se fasse qu'au lancement réel de la démo (et pas à l'import du module/tests).
    """
    import time

    from s2s_demo import VllmBackend, _wave

    class VllmToolBackend(VllmBackend):
        def __init__(self, ckpt: str, deploy: Path | None = None) -> None:
            super().__init__(ckpt, deploy_config=deploy)
            from vllm_omni_lfm2_audio.constants import IM_END_TOKEN_ID

            ids = self.tok.encode("<|tool_call_end|>", add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"<|tool_call_end|> doit être un token unique, obtenu {ids}")
            self.tool_call_end_id = ids[0]
            self.im_end_id = IM_END_TOKEN_ID
            self.last_text = ""

        def _render_turns(self, turns):
            """Tours → (prompt_token_ids, audio courant). 1 placeholder audio max
            (porté par le tour user de Pass A, en dernière position avant assistant)."""
            from vllm_omni_lfm2_audio.constants import AUDIO_FRAME_PLACEHOLDER_ID

            s = f"<|startoftext|><|im_start|>system\n{self.system}<|im_end|>\n"
            mm_audio = None
            for t in turns:
                text = t.text or ""
                if t.audio is not None:
                    audio_tok = self.tok.decode([AUDIO_FRAME_PLACEHOLDER_ID])
                    text = f"{audio_tok}{text}"
                    mm_audio = t.audio
                s += f"<|im_start|>{t.role}\n{text}<|im_end|>\n"
            s += "<|im_start|>assistant\n"
            return self.tok(s, add_special_tokens=False).input_ids, mm_audio

        def stream(self, turns, *, stop_token_ids):
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind

            token_ids, mm_audio = self._render_turns(turns)
            prompt: dict = {"prompt_token_ids": token_ids}
            if mm_audio is not None:
                wave, sr = mm_audio
                wave = np.asarray(wave, dtype=np.float32).reshape(-1)
                if sr != 16_000:  # le mel du conformer est calibré 16 kHz
                    import torch
                    import torchaudio

                    wave = torchaudio.functional.resample(torch.from_numpy(wave), sr, 16_000).numpy()
                    sr = 16_000
                prompt["multi_modal_data"] = {"audio": [(wave, sr)]}

            sp_pair = [
                SamplingParams(temperature=0.0, max_tokens=400, stop_token_ids=list(stop_token_ids),
                               output_kind=RequestOutputKind.FINAL_ONLY),
                SamplingParams(max_tokens=1, detokenize=False, output_kind=RequestOutputKind.DELTA),
            ]
            txt = ""
            _ = time.time()
            for o in self.omni.generate(prompt, sp_pair, py_generator=True, use_tqdm=False):
                ro = o.request_output
                if o.final_output_type == "text" and ro and ro.outputs:
                    txt = ro.outputs[0].text or txt
                elif o.final_output_type == "audio":
                    w = _wave(getattr(o, "multimodal_output", None) or getattr(ro, "multimodal_output", None))
                    if w is not None and w.size:
                        yield w
            self.last_text = (txt or "").strip()

    return VllmToolBackend(checkpoint, deploy_config)


# ──────────────────────────────── construction ───────────────────────────────


def build_agent(checkpoint: str, deploy_config: Path | None = None):
    from s2s_toolcalling.orchestrator.fillers import EN_FILLER_PHRASES, FillerBank
    from s2s_toolcalling.orchestrator.vllm_tool_agent import VllmToolAgent
    from s2s_toolcalling.tools.fake_db import FakeDbBackend
    from s2s_toolcalling.tools.toolcalling_en import build_toolcalling_en_registry
    from s2s_toolcalling.tools.web_search import DuckDuckGoBackend, TavilyBackend

    backend = _build_tool_backend(checkpoint, deploy_config)
    web = TavilyBackend(max_results=4) if os.environ.get("TAVILY_API_KEY") else DuckDuckGoBackend(max_results=4)
    print(f"[web] {'Tavily' if os.environ.get('TAVILY_API_KEY') else 'DuckDuckGo (repli)'}", flush=True)
    registry = build_toolcalling_en_registry(web_backend=web, db_backend=FakeDbBackend())
    # Pas de wav de filler ici (vLLM est rapide ; round-trip outil court) : on
    # n'émet que le TEXTE du filler dans l'UI. Filler vocal possible plus tard via
    # un checkpoint de base (« Perform TTS. » est OOD sur le modèle tool-calling).
    bank = FillerBank(phrases=dict(EN_FILLER_PHRASES))
    return VllmToolAgent(backend, registry, fillers=bank)


def build_stream(agent, turn: str):
    import gradio as gr
    from fastrtc import AdditionalOutputs, ReplyOnPause, Stream

    from s2s_toolcalling.orchestrator.events import (
        AudioChunk,
        FillerSpeech,
        ToolCallBegin,
        ToolCallResult,
        TurnComplete,
    )

    def handler(audio: tuple[int, np.ndarray]):
        sr, pcm = audio
        pcm = np.asarray(pcm)
        wave = (pcm.astype(np.float32).reshape(-1) / 32_768.0) if np.issubdtype(pcm.dtype, np.integer) \
            else pcm.astype(np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(wave**2))) if wave.size else 0.0
        print(f"👤 entrée {wave.size/sr:.1f}s @ {sr}Hz · RMS {rms:.3f}", flush=True)
        if rms < MIN_INPUT_RMS:
            print(f"   (ignoré : RMS {rms:.3f} < {MIN_INPUT_RMS} — écho/silence)", flush=True)
            return
        with LOCK:
            for ev in agent.respond(wave, sr):
                if isinstance(ev, ToolCallBegin):
                    line = f"🔧 {ev.name}({ev.arguments})"
                    print(line, flush=True)
                    yield AdditionalOutputs(line)
                elif isinstance(ev, ToolCallResult):
                    print(f"   ↳ outil ok={ev.ok} ({ev.elapsed_ms:.0f}ms) → {str(ev.payload)[:300]}", flush=True)
                    yield AdditionalOutputs(f"   ↳ {str(ev.payload)[:140]}")
                elif isinstance(ev, FillerSpeech):
                    yield AdditionalOutputs(f"💬 {ev.phrase}")
                elif isinstance(ev, AudioChunk):
                    # vLLM RTF<1 → on stream EN DIRECT (pas d'underrun, contrairement
                    # à liquid-audio) : c'est tout l'intérêt du port.
                    w = np.asarray(ev.samples, dtype=np.float32).reshape(-1)
                    pcm16 = (np.clip(w, -1.0, 1.0) * 32_767).astype(np.int16)
                    yield (SR_OUT, pcm16.reshape(1, -1))
                elif isinstance(ev, TurnComplete):
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

    reply_kwargs: dict = {"can_interrupt": False, "output_sample_rate": SR_OUT}
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
    ap.add_argument("--checkpoint", required=True, help="checkpoint Phase B CONVERTI (format omni, cf. docstring)")
    ap.add_argument("--deploy-config", type=Path, default=None, help="YAML d'optimisation (CUDA graphs) → TTFA ~250 ms")
    ap.add_argument("--turn", choices=["auto", "cloudflare", "none"], default="auto")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    turn = args.turn
    if turn == "auto":
        turn = "cloudflare" if (os.environ.get("CLOUDFLARE_TURN_KEY_ID") and
                                os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN")) else "none"

    agent = build_agent(args.checkpoint, args.deploy_config)
    stream = build_stream(agent, turn)
    print(f"\n▶ démo S2S + tool calling vLLM-Omni prête (TURN: {turn})", flush=True)
    stream.ui.launch(server_port=args.port, share=args.share, quiet=True)


if __name__ == "__main__":
    main()
