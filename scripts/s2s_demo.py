#!/usr/bin/env python3
"""Démo speech-to-speech LFM2.5-Audio — backend au choix : vLLM-Omni ou liquid-audio.

Entrée : audio (WAV) et/ou texte ; sortie : texte + WAV de la réponse.

Usage :
    # liquid-audio (référence) : speech → speech
    python scripts/s2s_demo.py --backend liquid --audio-in question.wav

    # vLLM-Omni : texte → speech (l'entrée audio n'est pas encore câblée côté vLLM)
    python scripts/s2s_demo.py --backend vllm --text "Hello, who are you?"

    # mode interactif (multi-tours) : tape du texte, ou `@/chemin/audio.wav`
    python scripts/s2s_demo.py --backend liquid --interactive

Limitation actuelle : le plugin vLLM expose ``embed_multimodal`` (conformer)
mais aucun processor multimodal n'est enregistré auprès de vLLM — l'entrée
audio du backend vllm sera disponible quand ce câblage sera fait (cf. docs).
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
        t0, ttfa = time.time(), None
        with torch.no_grad():
            for t in self.model.generate_interleaved(**self.chat, max_new_tokens=max_new_tokens):
                if t.numel() == 1:
                    text_ids.append(t.detach().cpu())
                else:
                    ttfa = ttfa or (time.time() - t0)
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
        return txt, wav, {"ttfa_s": ttfa, "total_s": total}


# ───────────────────────────── backend vLLM-Omni ─────────────────────────────


class VllmBackend:
    """vLLM-Omni : texte → speech (entrée audio : à venir, cf. docstring module)."""

    name = "vllm-omni"

    def __init__(self, checkpoint: str) -> None:
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
        self.omni = Omni(
            model=checkpoint,
            enforce_eager=True,
            gpu_memory_utilization=0.42,
            dtype="bfloat16",
            async_scheduling=False,
            async_chunk=True,
            stage_init_timeout=1200,
            init_timeout=1800,
        )
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.sp_pair = [
            SamplingParams(temperature=0.0, max_tokens=400, stop_token_ids=[IM_END_TOKEN_ID]),
            SamplingParams(max_tokens=1, detokenize=False),
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
        if audio_path is not None:
            raise NotImplementedError(
                "entrée audio non câblée pour le backend vllm — utilisez --text, "
                "ou --backend liquid pour le speech-to-speech complet"
            )
        if not text:
            raise ValueError("--text requis avec le backend vllm")
        self.history.append(("user", text))

        t0 = time.time()
        outs = self.omni.generate({"prompt_token_ids": self._render()}, self.sp_pair, use_tqdm=False)
        total = time.time() - t0

        txt, wav = "", None
        for o in outs:
            ro = o.request_output
            if o.final_output_type == "text" and ro and ro.outputs:
                txt = (ro.outputs[0].text or "").strip()
            elif o.final_output_type == "audio":
                mm = getattr(o, "multimodal_output", None) or getattr(ro, "multimodal_output", None)
                if isinstance(mm, dict):
                    v = mm.get("model_outputs")
                    if isinstance(v, torch.Tensor):
                        wav = v.detach().float().cpu().numpy().reshape(-1)
                    elif isinstance(v, np.ndarray):
                        wav = v.reshape(-1)
        self.history.append(("assistant", txt))
        return txt, wav, {"ttfa_s": None, "total_s": total}


# ──────────────────────────────────── main ───────────────────────────────────


def run_turn(backend, text, audio_in, out_path: Path) -> None:
    txt, wav, m = backend.reply(text=text, audio_path=audio_in)
    print(f"\n🤖 {txt}")
    if wav is not None and wav.size:
        save_wav(wav, out_path)
        dur = wav.size / SR_OUT
        ttfa = f"TTFA={m['ttfa_s']:.2f}s  " if m.get("ttfa_s") else ""
        print(f"🔊 {out_path}  ({dur:.1f}s d'audio — {ttfa}total={m['total_s']:.2f}s, "
              f"RTF={m['total_s']/dur:.2f})")
    else:
        print("(pas d'audio généré)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["liquid", "vllm"], required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="défaut : /workspace/models/LFM2.5-Audio-1.5B (liquid) ou "
                         "/workspace/models/lfm25_audio_omni (vllm)")
    ap.add_argument("--audio-in", type=Path, default=None, help="WAV d'entrée (speech)")
    ap.add_argument("--text", default=None, help="texte d'entrée (alternative ou complément)")
    ap.add_argument("--out", type=Path, default=Path("/workspace/audio_out/demo_reply.wav"))
    ap.add_argument("--interactive", action="store_true",
                    help="boucle multi-tours ; tape du texte ou `@/chemin/audio.wav`")
    args = ap.parse_args()

    if args.backend == "liquid":
        ckpt = args.checkpoint or "/workspace/models/LFM2.5-Audio-1.5B"
        backend = LiquidBackend(ckpt)
    else:
        ckpt = args.checkpoint or "/workspace/models/lfm25_audio_omni"
        backend = VllmBackend(ckpt)

    if not args.interactive:
        if args.text is None and args.audio_in is None:
            ap.error("--text et/ou --audio-in requis (ou --interactive)")
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
