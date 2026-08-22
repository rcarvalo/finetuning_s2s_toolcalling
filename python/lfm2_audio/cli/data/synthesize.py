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

    lfm2-synthesize-audio --engine voxtral \
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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf
import torch
import torchaudio
from kokoro import KPipeline

SAMPLE_RATE = 16_000
# Voix Kokoro (repli) — am/af = American male/female, bm/bf = British.
KOKORO_VOICES_TRAIN = ["af_heart", "af_bella", "am_adam", "am_michael", "bf_emma", "bm_george", "af_nicole", "am_eric"]
KOKORO_VOICES_HELDOUT = ["af_sarah", "bm_lewis"]
VOXTRAL_MODEL = "mistralai/Voxtral-4B-TTS-2603"


class VoxtralTTS:
    """Client du serveur Voxtral TTS (vLLM-Omni, endpoint OpenAI /v1/audio/speech)."""

    def __init__(
        self, base_url: str = "http://localhost:8000/v1", model: str = VOXTRAL_MODEL, timeout: float = 120.0
    ) -> None:
        self.url = base_url.rstrip("/") + "/audio/speech"
        self.model = model
        self.timeout = timeout

    def synth(self, text: str, voice: str) -> np.ndarray:

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

        self._pipeline = KPipeline(lang_code=lang_code)

    def synth(self, text: str, voice: str) -> np.ndarray:
        chunks = [audio for _, _, audio in self._pipeline(text, voice=voice)]
        wav24 = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(1, np.float32)
        return _resample(wav24, 24_000, SAMPLE_RATE)


def _resample(wav: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return wav

    return torchaudio.functional.resample(torch.from_numpy(wav), src, dst).numpy()


def _augment(wav: np.ndarray, rng: random.Random) -> np.ndarray:
    """Aug légère : variation de vitesse ±10 % + bruit blanc faible (robustesse)."""
    rate = 1.0 + rng.uniform(-0.1, 0.1)
    idx = np.clip((np.arange(0, len(wav), rate)).astype(int), 0, len(wav) - 1)
    wav = wav[idx]
    noise = np.random.default_rng(rng.randint(0, 2**31)).standard_normal(len(wav)).astype(np.float32)
    return np.clip(wav + rng.uniform(0.0, 0.005) * noise, -1.0, 1.0)


def _save_wav(wav: np.ndarray, path: Path) -> None:

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


def _build_engine(args: argparse.Namespace) -> Any:
    if args.engine == "voxtral":
        return VoxtralTTS(base_url=args.base_url)
    return KokoroTTS(lang_code=args.lang)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialogues", required=True, type=Path)
    ap.add_argument("--audio-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--engine", choices=["voxtral", "kokoro"], default="voxtral")
    ap.add_argument("--voices", default="", help="voix UTILISATEUR, séparées par des virgules (requis voxtral)")
    ap.add_argument(
        "--assistant-voice",
        default="",
        help="Phase B (S2S) : voix FIXE de l'assistant pour ses réponses parlées. "
        "Vide = pas de synthèse des tours assistant (Phase A single-turn).",
    )
    ap.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="train = voix d'entraînement ; test = voix held-out (inconnues)",
    )
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="serveur Voxtral (vLLM)")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="requêtes TTS en parallèle (voxtral ; vLLM les batch). Kokoro forcé à 1.",
    )
    ap.add_argument("--augment-prob", type=float, default=0.3)
    ap.add_argument("--lang", default="a", help="Kokoro lang_code (a=US, b=UK)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    voices = _resolve_voices(args)
    tts = _build_engine(args)

    with args.dialogues.open(encoding="utf-8") as handle:
        dialogues = [json.loads(line) for line in map(str.strip, handle) if line]

    # 1) construit les jobs (voix + décision d'aug tirées DANS le thread principal
    #    → déterministe), puis synthétise EN PARALLÈLE (le serveur vLLM batch).
    # Tours USER → voix utilisateur (variées, augmentées). Tours ASSISTANT parlés
    # (texte, pas de tool_calls) → voix assistant FIXE, sans aug (persona stable) ;
    # uniquement si --assistant-voice est fourni (Phase B S2S). Les tours tool-call
    # et tool (résultat) restent texte seul (pas d'audio).
    jobs = []  # (di, ti, text, voice, augment, rel)
    for di, dlg in enumerate(dialogues):
        for ti, turn in enumerate(dlg.get("turns", [])):
            if turn.get("audio") or not turn.get("text"):
                continue
            role = turn.get("role")
            if role == "user":
                voice, aug, tag = rng.choice(voices), rng.random() < args.augment_prob, "u"
            elif role == "assistant" and not turn.get("tool_calls") and args.assistant_voice:
                voice, aug, tag = args.assistant_voice, False, "a"
            else:
                continue
            jobs.append((di, ti, turn["text"], voice, aug, f"{dlg['id']}_{tag}{ti}.wav"))

    def _do(job: Any) -> Any:
        di, ti, text, voice, aug, rel = job
        try:
            wav = tts.synth(text, voice)
        except Exception as e:
            return job, None, str(e)
        if aug:
            wav = _augment(wav, random.Random(di * 1000 + ti))
        _save_wav(wav, args.audio_root / rel)
        return job, rel, None

    workers = args.concurrency if args.engine == "voxtral" else 1
    print(f"TTS {len(jobs)} utterances · {args.engine} · {workers} en parallèle · voix={voices}", flush=True)
    t0, failed, ok_di = time.time(), 0, set(range(len(dialogues)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, (job, rel, err) in enumerate(ex.map(_do, jobs), 1):
            di, ti, _, voice, _, _ = job
            if err:
                failed += 1
                ok_di.discard(di)
                print(f"[échec TTS] {job[2][:40]!r}: {err}", flush=True)
            else:
                dialogues[di]["turns"][ti]["audio"] = rel
                dialogues[di]["turns"][ti]["voice"] = voice
            if k % 50 == 0 or k == len(jobs):
                rate = k / (time.time() - t0)
                eta = (len(jobs) - k) / rate / 60
                print(f"  {k}/{len(jobs)} · {rate:.1f}/s · ETA {eta:.1f} min · échecs {failed}", flush=True)

    # 2) écrit les dialogues dont l'audio a réussi
    written = 0
    with args.out.open("w", encoding="utf-8") as fout:
        for di, dlg in enumerate(dialogues):
            if di in ok_di:
                fout.write(json.dumps(dlg, ensure_ascii=False) + "\n")
                written += 1
    print(
        f"\nsynthétisé {len(jobs) - failed} utterances → {written} dialogues écrits dans {args.out} "
        f"({failed} échecs) en {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
