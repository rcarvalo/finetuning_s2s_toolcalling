#!/usr/bin/env python3
"""TTS des tours user d'un JSONL de dialogues (capacité tool-calling EN).

Lit un JSONL au ``dialogue_schema`` où les tours user ont du ``text``, synthétise
chaque utterance, écrit les WAV 16 kHz mono et réécrit le JSONL avec le champ
``audio`` (l'adapter liquid préfère l'audio au texte). Voix tirée aléatoirement
par utterance ; un sous-ensemble de voix est RÉSERVÉ au test (généralisation à
des voix inconnues).

Deux moteurs (``--engine``) :
- ``voxtral`` (défaut) : Voxtral TTS servi par vLLM-Omni
  (``vllm serve mistralai/Voxtral-4B-TTS-2603 --omni``), endpoint OpenAI
  ``/v1/audio/speech`` — qualité nettement supérieure. Sur L4 (≥16 Go). Voix à
  passer via ``--voices`` (cf. cellule du notebook qui les liste).
- ``kokoro`` : repli local léger (Apache-2.0).

    python scripts/synthesize_user_audio.py --engine voxtral \
        --dialogues data/tc_en_train.jsonl --audio-root data/audio_tc_en \
        --out data/tc_en_train.audio.jsonl --voices casual_male,casual_female,...

Le TTS ne sert qu'à FABRIQUER les données hors-ligne : l'inférence reste un seul
modèle (LFM2.5-Audio). Sortie 16 kHz = rate du conformer (preprocessor).
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
# Voix Kokoro (repli) — am/af = American male/female, bm/bf = British.
KOKORO_VOICES_TRAIN = ["af_heart", "af_bella", "am_adam", "am_michael", "bf_emma", "bm_george", "af_nicole", "am_eric"]
KOKORO_VOICES_HELDOUT = ["af_sarah", "bm_lewis"]
VOXTRAL_MODEL = "mistralai/Voxtral-4B-TTS-2603"


class VoxtralTTS:
    """Client du serveur Voxtral TTS (vLLM-Omni, endpoint OpenAI /v1/audio/speech)."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = VOXTRAL_MODEL,
                 timeout: float = 120.0) -> None:
        self.url = base_url.rstrip("/") + "/audio/speech"
        self.model = model
        self.timeout = timeout

    def synth(self, text: str, voice: str) -> np.ndarray:
        import httpx
        import soundfile as sf

        payload = {"input": text, "model": self.model, "response_format": "wav", "voice": voice}
        resp = httpx.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        wav, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return _resample(wav.astype(np.float32), sr, SAMPLE_RATE)


class KokoroTTS:
    """Wrapper paresseux sur Kokoro (repli, 24 kHz natif → resample 16 kHz)."""

    def __init__(self, lang_code: str = "a") -> None:
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code=lang_code)

    def synth(self, text: str, voice: str) -> np.ndarray:
        chunks = [audio for _, _, audio in self._pipeline(text, voice=voice)]
        wav24 = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(1, np.float32)
        return _resample(wav24, 24_000, SAMPLE_RATE)


def _resample(wav: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return wav
    import torch
    import torchaudio

    return torchaudio.functional.resample(torch.from_numpy(wav), src, dst).numpy()


def _augment(wav: np.ndarray, rng: random.Random) -> np.ndarray:
    """Aug légère : variation de vitesse ±10 % + bruit blanc faible (robustesse)."""
    rate = 1.0 + rng.uniform(-0.1, 0.1)
    idx = np.clip((np.arange(0, len(wav), rate)).astype(int), 0, len(wav) - 1)
    wav = wav[idx]
    noise = np.random.default_rng(rng.randint(0, 2**31)).standard_normal(len(wav)).astype(np.float32)
    return np.clip(wav + rng.uniform(0.0, 0.005) * noise, -1.0, 1.0)


def _save_wav(wav: np.ndarray, path: Path) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, SAMPLE_RATE, subtype="PCM_16")


def _resolve_voices(args: argparse.Namespace) -> list[str]:
    """Voix à utiliser selon le moteur et le split (train vs held-out)."""
    explicit = [v.strip() for v in (args.voices or "").split(",") if v.strip()]
    if args.engine == "voxtral":
        if not explicit:
            raise SystemExit(
                "Voxtral : précise --voices (liste séparée par des virgules). "
                "Le notebook fournit une cellule qui liste les 20 voix disponibles."
            )
        return explicit
    # kokoro : défauts par split, surchargés par --voices si fourni
    if explicit:
        return explicit
    return KOKORO_VOICES_HELDOUT if args.split == "test" else KOKORO_VOICES_TRAIN


def _build_engine(args: argparse.Namespace):
    if args.engine == "voxtral":
        return VoxtralTTS(base_url=args.base_url)
    return KokoroTTS(lang_code=args.lang)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialogues", required=True, type=Path)
    ap.add_argument("--audio-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--engine", choices=["voxtral", "kokoro"], default="voxtral")
    ap.add_argument("--voices", default="", help="voix séparées par des virgules (requis pour voxtral)")
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="train = voix d'entraînement ; test = voix held-out (inconnues)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="serveur Voxtral (vLLM)")
    ap.add_argument("--augment-prob", type=float, default=0.3)
    ap.add_argument("--lang", default="a", help="Kokoro lang_code (a=US, b=UK)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    voices = _resolve_voices(args)
    tts = _build_engine(args)

    n_utt = 0
    with args.dialogues.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            dlg = json.loads(line)
            for i, turn in enumerate(dlg.get("turns", [])):
                if turn.get("role") != "user" or not turn.get("text") or turn.get("audio"):
                    continue
                voice = rng.choice(voices)
                wav = tts.synth(turn["text"], voice)
                if rng.random() < args.augment_prob:
                    wav = _augment(wav, rng)
                rel = f"{dlg['id']}_u{i}.wav"
                _save_wav(wav, args.audio_root / rel)
                turn["audio"] = rel
                turn["voice"] = voice
                n_utt += 1
            fout.write(json.dumps(dlg, ensure_ascii=False) + "\n")

    print(f"synthétisé {n_utt} utterances ({args.engine}, {args.split}, voix={voices}) → {args.out}")


if __name__ == "__main__":
    main()
