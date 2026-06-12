#!/usr/bin/env python3
"""Démo speech-to-speech LFM2.5-Audio — backend au choix : vLLM-Omni ou liquid-audio.

Entrée : audio (WAV) et/ou texte ; sortie : texte + WAV de la réponse.

Usage :
    # liquid-audio (référence) : speech → speech
    python scripts/s2s_demo.py --backend liquid --audio-in question.wav

    # vLLM-Omni : speech → speech (audio-in natif) ou texte → speech
    python scripts/s2s_demo.py --backend vllm --audio-in question.wav
    python scripts/s2s_demo.py --backend vllm --text "Hello, who are you?"

    # mode interactif (multi-tours) : tape du texte, ou `@/chemin/audio.wav`
    python scripts/s2s_demo.py --backend liquid --interactive

Backend vllm : streaming in-process py_generator (recette bench_ttfa /
s2s_webrtc — stage 1 en DELTA ; le chemin non-streaming FINAL_ONLY ne sort
aucun chunk audio), chunks concaténés en un WAV final.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SR_OUT = 24_000
SYSTEM = "Respond with interleaved text and audio."
END_OF_AUDIO_CODE = 2048


def _wave(x) -> np.ndarray | None:
    """Extrait un waveform du multimodal_output (clé "audio" en DELTA,
    "model_outputs" en FINAL_ONLY) — même contrat que bench_ttfa."""
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("model_outputs", "audio", "waveform", "wav"):
            if k in x:
                return _wave(x[k])
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy().reshape(-1)
    if isinstance(x, np.ndarray):
        return x.reshape(-1)
    return None


def save_wav(wav: np.ndarray, path: Path, rate: int = SR_OUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32_767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(pcm16.tobytes())


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    import soundfile as sf
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T[:1]), rate  # (1, T) mono


# ───────────────────────────── backend liquid-audio ──────────────────────────


class LiquidBackend:
    """Référence liquid-audio : speech → speech natif, batch=1."""

    name = "liquid-audio"

    def __init__(self, checkpoint: str) -> None:
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
        t0 = time.time()
        src = Path(checkpoint) if Path(checkpoint).exists() else checkpoint
        self.model = LFM2AudioModel.from_pretrained(src, device="cuda").eval()
        self.proc = LFM2AudioProcessor.from_pretrained(src, device="cuda")
        self._ChatState = ChatState
        self.chat = None
        self.reset()
        print(f"[{self.name}] prêt en {time.time()-t0:.0f}s")

    def reset(self) -> None:
        self.chat = self._ChatState(self.proc)
        self.chat.new_turn("system"); self.chat.add_text(SYSTEM); self.chat.end_turn()

    def reply(self, text: str | None = None, audio_path: Path | None = None,
              max_new_tokens: int = 512) -> tuple[str, np.ndarray | None, dict]:
        self.chat.new_turn("user")
        if audio_path is not None:
            wav_in, rate = load_audio(audio_path)
            self.chat.add_audio(wav_in, rate)
        if text:
            self.chat.add_text(text)
        self.chat.end_turn()
        self.chat.new_turn("assistant")

        text_ids, frames = [], []
        t0, first_frame = time.time(), None
        with torch.no_grad():
            # audio_temperature/audio_top_k : valeurs de l'exemple officiel
            # LFM2.5-Audio (parité avec le sampling audio du plugin vLLM)
            for t in self.model.generate_interleaved(
                **self.chat, max_new_tokens=max_new_tokens,
                audio_temperature=1.0, audio_top_k=4,
            ):
                if t.numel() == 1:
                    text_ids.append(t.detach().cpu())
                else:
                    first_frame = first_frame or (time.time() - t0)
                    frames.append(t.detach().cpu())
        txt = self.proc.text.decode([int(x) for x in text_ids]).replace("<|text_end|>", "").strip()
        # réinjecte le texte de la réponse dans l'historique pour le tour
        # suivant (les frames audio sont omises, comme côté vLLM)
        self.chat.add_text(txt)
        self.chat.end_turn()
        keep = [f.flatten() for f in frames if int(f.flatten()[0]) != END_OF_AUDIO_CODE]
        wav = None
        if keep:
            with torch.no_grad():
                w = self.proc.decode(torch.stack(keep, dim=1).cuda().unsqueeze(0))
            wav = w.float().cpu().numpy().reshape(-1)
        total = time.time() - t0
        # TTFA = premier audio AUDIBLE (convention bench_ttfa) : liquid décode
        # Mimi en bloc à la fin → rien n'est jouable avant `total`. Le premier
        # token audio (non audible) est reporté à part (first_frame_s).
        return txt, wav, {"ttfa_s": total if wav is not None else None,
                          "first_frame_s": first_frame, "total_s": total}


# ───────────────────────────── backend vLLM-Omni ─────────────────────────────


class VllmBackend:
    """vLLM-Omni : texte → speech (entrée audio : à venir, cf. docstring module)."""

    name = "vllm-omni"

    def __init__(self, checkpoint: str, deploy_config: Path | None = None) -> None:
        sys.path.insert(0, str(REPO / "src"))
        import vllm_omni.plugins as _p
        _p.omni_plugins_loaded = False
        import vllm_omni_lfm2_audio  # noqa: F401
        from vllm_omni.plugins import load_omni_general_plugins
        load_omni_general_plugins()
        from vllm import SamplingParams
        from vllm_omni import Omni
        from transformers import AutoTokenizer
        from vllm_omni_lfm2_audio.constants import IM_END_TOKEN_ID

        t0 = time.time()
        kwargs: dict = dict(model=checkpoint, async_chunk=True,
                            stage_init_timeout=1200, init_timeout=1800)
        if deploy_config:
            # chemin OPTIMISÉ (même recette que bench_ttfa) : CUDA graphs
            # PIECEWISE stage 0 + initial_codec_chunk_frames=2 → TTFA visé
            # 250-350 ms (cf. docs/optimization_audit.md) ; eager ≈ 750 ms.
            kwargs["deploy_config"] = str(deploy_config)
        else:
            kwargs.update(
                enforce_eager=True,
                gpu_memory_utilization=0.42,
                dtype="bfloat16",
                async_scheduling=False,
                # APC actif par défaut (vLLM 0.22) → omni_prefix_cache du runner
                # PERD l'export sparse codes.audio → texte OK mais ZÉRO chunk
                # vers le stage 1 (mesuré 12/06, cf. le YAML).
                enable_prefix_caching=False,
            )
        self.omni = Omni(**kwargs)
        # Streaming in-process — recette VALIDÉE (bench_ttfa / s2s_webrtc) :
        # Omni.generate force FINAL_ONLY sur les stages LLM → aucun chunk audio
        # ne sort (ni en cours ni en fin de génération sur ce chemin) ; on
        # neutralise pour respecter le DELTA du stage 1, et py_generator ne
        # doit pas fermer l'engine à l'épuisement (multi-tours).
        self.omni._set_final_only_for_llm_stages = lambda spl: list(spl)
        self.omni._real_close, self.omni.close = self.omni.close, lambda: None

        from vllm.sampling_params import RequestOutputKind

        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.sp_pair = [
            SamplingParams(temperature=0.0, max_tokens=400, stop_token_ids=[IM_END_TOKEN_ID],
                           output_kind=RequestOutputKind.FINAL_ONLY),
            SamplingParams(max_tokens=1, detokenize=False,
                           output_kind=RequestOutputKind.DELTA),
        ]
        self.history: list[tuple[str, str]] = []
        print(f"[{self.name}] prêt en {time.time()-t0:.0f}s")

    def reset(self) -> None:
        self.history.clear()

    def _render(self) -> list[int]:
        s = f"<|startoftext|><|im_start|>system\n{SYSTEM}<|im_end|>\n"
        for role, txt in self.history:
            s += f"<|im_start|>{role}\n{txt}<|im_end|>\n"
        return self.tok(s + "<|im_start|>assistant\n", add_special_tokens=False).input_ids

    def reply(self, text: str | None = None, audio_path: Path | None = None,
              max_new_tokens: int = 400) -> tuple[str, np.ndarray | None, dict]:
        multi_modal_data = None
        if audio_path is not None:
            # Audio-in NATIF : 1 token audio_in dans le tour user, remplacé par
            # ceil(T_mel/8) placeholders par le processor mm ; les embeddings
            # conformer sont mergés au prefill (cf. docs/audio_in_spec.md).
            import soundfile as sf

            wave, sr = sf.read(str(audio_path), dtype="float32")
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            multi_modal_data = {"audio": [(wave, sr)]}
            from vllm_omni_lfm2_audio.constants import AUDIO_FRAME_PLACEHOLDER_ID

            audio_tok = self.tok.decode([AUDIO_FRAME_PLACEHOLDER_ID])
            text = f"{audio_tok}{text or ''}"
        if not text:
            raise ValueError("--text et/ou --audio-in requis avec le backend vllm")
        self.history.append(("user", text))

        t0 = time.time()
        prompt: dict = {"prompt_token_ids": self._render()}
        if multi_modal_data is not None:
            prompt["multi_modal_data"] = multi_modal_data

        from vllm_omni_lfm2_audio.constants import (
            AUDIO_EOA_PLACEHOLDER_ID,
            AUDIO_FRAME_PLACEHOLDER_ID,
        )

        txt, chunks, frames, ttfa = "", [], 0, None
        for o in self.omni.generate(prompt, self.sp_pair, py_generator=True, use_tqdm=False):
            ro = o.request_output
            if o.final_output_type == "text" and ro and ro.outputs:
                txt = (ro.outputs[0].text or txt).strip()
                # frames émises par le stage 0 : vérité terrain sur l'audio généré
                toks = list(ro.outputs[0].token_ids or [])
                frames = sum(1 for t in toks
                             if t in (AUDIO_FRAME_PLACEHOLDER_ID, AUDIO_EOA_PLACEHOLDER_ID))
            elif o.final_output_type == "audio":
                w = _wave(getattr(o, "multimodal_output", None) or getattr(ro, "multimodal_output", None))
                if w is not None and w.size:
                    ttfa = ttfa or (time.time() - t0)
                    chunks.append(w)
        total = time.time() - t0
        wav = np.concatenate(chunks) if chunks else None
        self.history.append(("assistant", txt))
        return txt, wav, {"ttfa_s": ttfa, "total_s": total, "frames": frames}


# ──────────────────────────────────── main ───────────────────────────────────


def run_turn(backend, text, audio_in, out_path: Path) -> None:
    txt, wav, m = backend.reply(text=text, audio_path=audio_in)
    print(f"\n🤖 {txt}")
    if wav is not None and wav.size:
        save_wav(wav, out_path)
        dur = wav.size / SR_OUT
        ttfa = f"TTFA={m['ttfa_s']:.2f}s  " if m.get("ttfa_s") else ""
        ff = f"1re frame={m['first_frame_s']:.2f}s  " if m.get("first_frame_s") else ""
        print(f"🔊 {out_path}  ({dur:.1f}s d'audio — {ttfa}{ff}total={m['total_s']:.2f}s, "
              f"RTF={m['total_s']/dur:.2f})")
    elif m.get("frames"):
        print(f"⚠️  {m['frames']} frames audio émises par le stage 0 mais aucun wav reçu "
              "du stage 1 → plomberie connector/stage 1 (cf. bench_ttfa --debug-chunk)")
    elif "frames" in m:
        print("⚠️  le stage 0 n'a émis AUCUNE frame audio (pas de <|text_end|>) "
              "→ prompting/modèle, pas de plomberie")
    else:
        print("(pas d'audio généré)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["liquid", "vllm"], required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="défaut : /workspace/models/LFM2.5-Audio-1.5B (liquid) ou "
                         "/workspace/models/lfm25_audio_omni (vllm)")
    ap.add_argument("--deploy-config", type=Path,
                    default=REPO / "configs/vllm_omni_lfm2_audio.yaml",
                    help="YAML de déploiement vLLM-Omni (CUDA graphs + chunks TTFA)")
    ap.add_argument("--no-deploy-config", action="store_true",
                    help="kwargs legacy (tout eager) au lieu du YAML")
    ap.add_argument("--audio-in", type=Path, default=None, help="WAV d'entrée (speech)")
    ap.add_argument("--text", default=None, help="texte d'entrée (alternative ou complément)")
    ap.add_argument("--out", type=Path, default=Path("/workspace/audio_out/demo_reply.wav"))
    ap.add_argument("--warmup", type=int, default=0,
                    help="tours de chauffe avant le tour mesuré (JIT Triton + "
                         "autotuning CUDA graphs : le 1er tour à froid coûte ~1 s de plus)")
    ap.add_argument("--interactive", action="store_true",
                    help="boucle multi-tours ; tape du texte ou `@/chemin/audio.wav`")
    args = ap.parse_args()

    if args.backend == "liquid":
        ckpt = args.checkpoint or "/workspace/models/LFM2.5-Audio-1.5B"
        backend = LiquidBackend(ckpt)
    else:
        ckpt = args.checkpoint or "/workspace/models/lfm25_audio_omni"
        deploy = None if args.no_deploy_config else args.deploy_config
        backend = VllmBackend(ckpt, deploy_config=deploy)

    if not args.interactive:
        if args.text is None and args.audio_in is None:
            ap.error("--text et/ou --audio-in requis (ou --interactive)")
        for i in range(args.warmup):
            t0 = time.time()
            backend.reply(text=args.text, audio_path=args.audio_in)
            backend.reset()
            print(f"[warmup {i+1}/{args.warmup}] {time.time()-t0:.2f}s")
        run_turn(backend, args.text, args.audio_in, args.out)
        return

    print("Mode interactif — texte, `@/chemin/audio.wav`, `/reset`, ou Ctrl-D pour quitter.")
    n = 0
    while True:
        try:
            line = input("\n👤 > ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == "/reset":
            backend.reset()
            print("(historique vidé)")
            continue
        text, audio_in = (None, Path(line[1:])) if line.startswith("@") else (line, None)
        out = args.out.with_name(f"{args.out.stem}_{n:02d}.wav")
        try:
            run_turn(backend, text, audio_in, out)
        except (NotImplementedError, ValueError, FileNotFoundError) as e:
            print(f"⚠️  {e}")
            continue
        n += 1


if __name__ == "__main__":
    main()
