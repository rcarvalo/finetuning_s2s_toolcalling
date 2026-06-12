#!/usr/bin/env python3
"""Conversation vocale temps réel (WebRTC) avec LFM2.5-Audio.

Tu parles dans le micro ; dès que tu fais une pause (VAD), la réponse vocale
commence à jouer PENDANT qu'elle est encore en train d'être générée :

  micro ──WebRTC──▶ VAD (fastrtc) ──▶ ASR (liquid-audio, ~1 s)
        ──▶ génération streamée ──▶ chunks de 0,8 s ──WebRTC──▶ haut-parleur

Backends de génération :
  --backend vllm    chunks streamés par le moteur vLLM-Omni (stage 1 en DELTA)
  --backend liquid  generate_interleaved + décodage Mimi par fenêtres de 10 frames

Lancement (pod) :   python scripts/s2s_webrtc_demo.py --backend vllm --port 7860
Accès (machine) :   ssh -L 7860:localhost:7860 root@<pod> → http://localhost:7860

L'entrée audio du plugin vLLM n'étant pas encore câblée, l'ASR passe par le
modèle liquid dans les deux modes (≈ 1 s) ; la génération utilise le backend choisi.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SYSTEM = "Respond with interleaved text and audio. Keep your spoken answers short and conversational."
TRANSCRIBE_SYSTEM = "Transcribe the user's speech verbatim. Respond with the transcription only, as text."
SR_OUT = 24_000
END_OF_AUDIO_CODE = 2048
CHUNK_FRAMES, LEFT_CTX = 10, 13  # mêmes fenêtres que le stage 1 vLLM
SAMPLES_PER_FRAME = 1920

LOCK = threading.Lock()


# ─────────────────────────── modèles ───────────────────────────


class Models:
    def __init__(self, backend: str) -> None:
        from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

        self._ChatState = ChatState
        self.backend = backend
        self.history: list[tuple[str, str]] = []

        self.omni = None
        if backend == "vllm":
            self._load_vllm()

        t0 = time.time()
        src = Path("/workspace/models/LFM2.5-Audio-1.5B")
        self.liquid = LFM2AudioModel.from_pretrained(src, device="cuda").eval()
        self.proc = LFM2AudioProcessor.from_pretrained(src, device="cuda")
        print(f"[liquid] prêt en {time.time()-t0:.0f}s (ASR{' + génération' if backend == 'liquid' else ''})")

    def _load_vllm(self) -> None:
        import vllm_omni.plugins as _p
        _p.omni_plugins_loaded = False
        import vllm_omni_lfm2_audio  # noqa: F401
        from vllm_omni.plugins import load_omni_general_plugins
        load_omni_general_plugins()
        from transformers import AutoTokenizer
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from vllm_omni import Omni
        from vllm_omni_lfm2_audio.constants import IM_END_TOKEN_ID

        ckpt = "/workspace/models/lfm25_audio_omni"
        t0 = time.time()
        self.omni = Omni(
            model=ckpt, enforce_eager=True, gpu_memory_utilization=0.42, dtype="bfloat16",
            async_scheduling=False, async_chunk=True, stage_init_timeout=1200, init_timeout=1800,
            # APC par défaut (vLLM 0.22) → perte de l'export codes.audio vers le
            # stage 1 (cf. configs/vllm_omni_lfm2_audio.yaml, mesuré 12/06)
            enable_prefix_caching=False,
        )
        # streaming : pas de FINAL_ONLY forcé, stage 1 en DELTA ; py_generator ne
        # doit pas fermer l'engine entre deux tours
        self.omni._set_final_only_for_llm_stages = lambda spl: list(spl)
        self.omni._real_close, self.omni.close = self.omni.close, lambda: None
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.sp_pair = [
            SamplingParams(temperature=0.0, max_tokens=400, stop_token_ids=[IM_END_TOKEN_ID],
                           output_kind=RequestOutputKind.FINAL_ONLY),
            SamplingParams(max_tokens=1, detokenize=False, output_kind=RequestOutputKind.DELTA),
        ]
        print(f"[vllm] prêt en {time.time()-t0:.0f}s")

    # ── ASR (liquid, tour isolé) ──

    def transcribe(self, wav_16k: torch.Tensor) -> str:
        chat = self._ChatState(self.proc)
        chat.new_turn("system"); chat.add_text(TRANSCRIBE_SYSTEM); chat.end_turn()
        chat.new_turn("user"); chat.add_audio(wav_16k, 16_000); chat.end_turn()
        chat.new_turn("assistant")
        ids = []
        with torch.no_grad():
            for t in self.liquid.generate_interleaved(**chat, max_new_tokens=128):
                if t.numel() == 1:
                    ids.append(int(t))
        return self.proc.text.decode(ids).replace("<|text_end|>", "").strip()

    # ── génération streamée : yield (np.float32 24 kHz) ──

    def stream_reply(self, user_text: str):
        gen = self._stream_vllm if self.backend == "vllm" else self._stream_liquid
        self.history.append(("user", user_text))
        text = yield from gen()
        self.history.append(("assistant", text))
        return text

    def _render_ids(self) -> list[int]:
        s = f"<|startoftext|><|im_start|>system\n{SYSTEM}<|im_end|>\n"
        for role, txt in self.history:
            s += f"<|im_start|>{role}\n{txt}<|im_end|>\n"
        return self.tok(s + "<|im_start|>assistant\n", add_special_tokens=False).input_ids

    def _stream_vllm(self):
        import os
        debug = os.environ.get("S2S_DEBUG")
        text = ""
        for o in self.omni.generate({"prompt_token_ids": self._render_ids()}, self.sp_pair,
                                    py_generator=True, use_tqdm=False):
            if debug:
                mm0 = getattr(o, "multimodal_output", None)
                mm1 = getattr(o.request_output, "multimodal_output", None) if o.request_output else None
                print(f"[dbg] type={o.final_output_type} stage={o.stage_id} "
                      f"mm_self={type(mm0).__name__}{list(mm0) if isinstance(mm0, dict) else ''} "
                      f"mm_ro={type(mm1).__name__}{list(mm1) if isinstance(mm1, dict) else ''}", flush=True)
            if o.final_output_type == "audio":
                mm = getattr(o, "multimodal_output", None) or getattr(o.request_output, "multimodal_output", None)
                if isinstance(mm, dict):
                    # clé "audio" en mode DELTA (chunk), "model_outputs" en FINAL_ONLY
                    v = next((mm[k] for k in ("audio", "model_outputs", "waveform", "wav") if k in mm), None)
                    if isinstance(v, torch.Tensor):
                        v = v.detach().float().cpu().numpy()
                    if isinstance(v, np.ndarray) and v.size:
                        yield v.astype(np.float32).reshape(-1)
            elif o.final_output_type == "text":
                ro = o.request_output
                if ro and ro.outputs:
                    text = (ro.outputs[0].text or text).strip()
        return text

    def _stream_liquid(self):
        chat = self._ChatState(self.proc)
        chat.new_turn("system"); chat.add_text(SYSTEM); chat.end_turn()
        for role, txt in self.history:
            chat.new_turn(role); chat.add_text(txt); chat.end_turn()
        chat.new_turn("assistant")

        text_ids: list[int] = []
        frames: list[torch.Tensor] = []
        emitted = 0

        def decode_new() -> np.ndarray:
            nonlocal emitted
            start = max(0, emitted - LEFT_CTX)
            window = torch.stack([f.flatten() for f in frames[start:]], dim=1)  # (8, T)
            with torch.no_grad():
                w = self.proc.decode(window.cuda().unsqueeze(0))
            cut = (emitted - start) * SAMPLES_PER_FRAME
            emitted = len(frames)
            return w.float().cpu().numpy().reshape(-1)[cut:]

        with torch.no_grad():
            for t in self.liquid.generate_interleaved(**chat, max_new_tokens=512):
                if t.numel() == 1:
                    text_ids.append(int(t))
                    continue
                f = t.detach().cpu().flatten()
                if int(f[0]) == END_OF_AUDIO_CODE:
                    continue
                frames.append(f)
                if len(frames) - emitted >= CHUNK_FRAMES:
                    yield decode_new()
        if len(frames) > emitted:
            yield decode_new()

        text = self.proc.text.decode(text_ids).replace("<|text_end|>", "").strip()
        chat.add_text(text); chat.end_turn()
        return text


# ─────────────────────────── fastrtc ───────────────────────────


def build_stream(models: Models):
    import torchaudio
    from fastrtc import AdditionalOutputs, ReplyOnPause, Stream

    def handler(audio: tuple[int, np.ndarray]):
        sr, pcm = audio
        wav = torch.from_numpy(np.asarray(pcm, dtype=np.float32).reshape(1, -1) / 32768.0)
        wav16 = torchaudio.functional.resample(wav, sr, 16_000)

        with LOCK:
            t0 = time.time()
            user_text = models.transcribe(wav16)
            t_asr = time.time() - t0
            print(f"\n👤 ({t_asr:.2f}s ASR) {user_text}")
            if not user_text:
                return
            yield AdditionalOutputs(f"👤 {user_text}")

            first, n, total = None, 0, 0
            gen = models.stream_reply(user_text)
            text = ""
            while True:
                try:
                    chunk = next(gen)
                except StopIteration as st:
                    text = st.value or ""
                    break
                first = first or (time.time() - t0)
                n += 1
                total += chunk.size
                pcm16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
                yield (SR_OUT, pcm16.reshape(1, -1))
            dt = time.time() - t0
            print(f"🤖 {text}")
            print(f"   [{models.backend}] 1er chunk {first:.2f}s après fin de parole "
                  f"(ASR {t_asr:.2f}s incluse) · {n} chunks · {total/SR_OUT:.1f}s d'audio · gen {dt:.1f}s")
            yield AdditionalOutputs(f"🤖 {text}\n⏱ 1er son: {first:.2f}s · {total/SR_OUT:.1f}s d'audio en {dt:.1f}s")

    import gradio as gr
    return Stream(
        handler=ReplyOnPause(handler, can_interrupt=False, output_sample_rate=SR_OUT),
        modality="audio",
        mode="send-receive",
        additional_outputs=[gr.Textbox(label="conversation")],
        additional_outputs_handler=lambda old, new: ((old + "\n") if old else "") + new,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "liquid"], default="vllm")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    models = Models(args.backend)
    stream = build_stream(models)
    print(f"\n▶ démo WebRTC prête ({args.backend}) : ssh -L {args.port}:localhost:{args.port} puis "
          f"http://localhost:{args.port}")
    stream.ui.launch(server_name="127.0.0.1", server_port=args.port, quiet=True)


if __name__ == "__main__":
    main()
