"""Analyse d'un dataset de dialogues : distribution, qualité texte, qualité audio.

Python pur (numpy/soundfile) : le rapport se calcule sans GPU et chaque
indicateur se teste isolément. La CLI correspondante est ``lfm2-analyze-dataset``.
"""

from __future__ import annotations

import json
import random
import re
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf

_PLACEHOLDER = re.compile(r"\[[^\]]+\]")
NEG_RATIO_OK = (0.20, 0.35)


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in map(str.strip, handle) if line]


def _utt(r: dict) -> str:
    return next((t.get("text", "") for t in r["turns"] if t.get("role") == "user"), "")


def _assistant(r: dict) -> dict:
    return next((t for t in r["turns"] if t.get("role") == "assistant"), {})


def _target(r: dict) -> str:
    if r.get("meta", {}).get("target"):
        return r["meta"]["target"]
    calls = _assistant(r).get("tool_calls", [])
    return calls[0]["name"] if calls else "none"


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def distribution(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    tgt = Counter(_target(r) for r in rows)
    neg = tgt.get("none", 0)
    return {
        "n": n,
        "targets": dict(tgt),
        "neg_ratio": neg / n if n else 0.0,
        "styles": dict(Counter(r.get("meta", {}).get("style") for r in rows)),
        "depths": dict(Counter(r.get("meta", {}).get("depth") for r in rows)),
    }


def text_quality(rows: list[dict]) -> dict[str, Any]:
    words = [float(len(_utt(r).split())) for r in rows]
    dups = sum(c - 1 for c in Counter(_norm(_utt(r)) for r in rows).values() if c > 1)
    placeholders = sum(1 for r in rows if _target(r) == "none" and _PLACEHOLDER.search(_assistant(r).get("text", "")))

    def _args(tool: str, key: str) -> list[str]:
        return [_assistant(r)["tool_calls"][0]["arguments"].get(key, "") for r in rows if _target(r) == tool]

    out: dict[str, Any] = {
        "utterance_words": _stat(words),
        "duplicate_utterances": dups,
        "placeholder_negatives": placeholders,
        "args": {},
    }
    for tool, key in (("web_search", "query"), ("db_query", "question")):
        vals = _args(tool, key)
        if vals:
            out["args"][tool] = {
                "count": len(vals),
                "empty": sum(1 for v in vals if not v.strip()),
                "words": _stat([float(len(v.split())) for v in vals]),
                "distinct_ratio": len({_norm(v) for v in vals}) / len(vals),
            }
    return out


def voices(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(t.get("voice") for r in rows for t in r["turns"] if t.get("role") == "user" and t.get("voice")))


def audio_quality(rows: list[dict], audio_root: str | Path, *, rms_sample: int = 150, seed: int = 0) -> dict[str, Any]:

    root = Path(audio_root)
    paths = [
        (r["id"], root / t["audio"]) for r in rows for t in r["turns"] if t.get("role") == "user" and t.get("audio")
    ]
    missing = [str(p) for _, p in paths if not p.exists()]
    durations: list[float] = []
    rates: Counter[int] = Counter()
    for _, p in paths:
        if p.exists():
            info = sf.info(str(p))
            durations.append(info.frames / info.samplerate)
            rates[info.samplerate] += 1
    # RMS sur un échantillon (lecture des samples = plus lourd)
    rng = random.Random(seed)
    sample = rng.sample([p for _, p in paths if p.exists()], min(rms_sample, len(durations)))
    silent = 0
    for p in sample:
        wav, _ = sf.read(str(p), dtype="float32")
        if float((wav**2).mean()) ** 0.5 < 0.005:
            silent += 1
    return {
        "files": len(paths),
        "missing": len(missing),
        "sample_rates": dict(rates),
        "duration_s": _stat(durations),
        "rms_checked": len(sample),
        "silent_in_sample": silent,
    }


def _stat(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": round(min(xs), 2),
        "p50": round(st.median(xs), 2),
        "max": round(max(xs), 2),
        "mean": round(st.mean(xs), 2),
    }


def analyze(rows: list[dict], audio_root: str | Path | None = None) -> dict[str, Any]:
    rep = {"distribution": distribution(rows), "text": text_quality(rows), "voices": voices(rows)}
    has_audio = any(t.get("audio") for r in rows for t in r["turns"])
    if audio_root and has_audio:
        rep["audio"] = audio_quality(rows, audio_root)
    return rep


def flags(rep: dict[str, Any]) -> list[str]:
    """Problèmes détectés (chaîne vide = dataset sain)."""
    out: list[str] = []
    d, t = rep["distribution"], rep["text"]
    if not NEG_RATIO_OK[0] <= d["neg_ratio"] <= NEG_RATIO_OK[1]:
        out.append(f"ratio négatifs {d['neg_ratio']:.0%} hors cible {NEG_RATIO_OK[0]:.0%}-{NEG_RATIO_OK[1]:.0%}")
    if t["duplicate_utterances"]:
        out.append(f"{t['duplicate_utterances']} utterances dupliquées")
    if t["placeholder_negatives"]:
        out.append(f"{t['placeholder_negatives']} négatifs à trous [placeholder]")
    for tool, a in t["args"].items():
        if a["empty"]:
            out.append(f"{tool}: {a['empty']} arguments vides")
        if a["distinct_ratio"] < 0.8:
            out.append(f"{tool}: faible diversité d'arguments ({a['distinct_ratio']:.0%} distincts)")
    if "audio" in rep:
        au = rep["audio"]
        if au["missing"]:
            out.append(f"{au['missing']} fichiers audio manquants")
        if au["silent_in_sample"]:
            out.append(f"{au['silent_in_sample']}/{au['rms_checked']} audios quasi-silencieux (échantillon)")
        if set(au["sample_rates"]) - {16000}:
            out.append(f"sample rates inattendus : {au['sample_rates']} (attendu 16000)")
    return out
