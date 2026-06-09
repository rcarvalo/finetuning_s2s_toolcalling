"""Calibre le ratio interleaved texte:audio pour le français (Phase 2a).

Principe (recette du variant JP, ratio 6:9 « based on tokenization statistics ») :
en mode interleaved, le modèle alterne n_text tokens texte et n_audio frames
Mimi (12,5 frames/s). Le texte doit rester EN AVANCE sur l'audio : il faut

    n_text / n_audio  >=  densité (tokens/s de parole) / 12,5

avec une marge de sécurité. Anglais : 6:12 → budget 6,25 tokens/s.

Entrée : JSONL ``{"text": "...", "duration_s": 4.2}`` (transcripts FR + durées,
p.ex. tirés de Common Voice FR). Sortie : densité mesurée et n_audio suggéré
pour n_text=6.

Usage :
    python scripts/calibrate_interleaved_ratio.py --manifest cv_fr_stats.jsonl \\
        [--model-id LiquidAI/LFM2.5-Audio-1.5B] [--margin 1.15]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MIMI_FRAME_RATE = 12.5  # frames/s
EN_RATIO = (6, 12)
JP_RATIO = (6, 9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSONL {text, duration_s}")
    parser.add_argument("--model-id", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--n-text", type=int, default=6)
    parser.add_argument("--margin", type=float, default=1.15, help="marge de sécurité sur la densité (p90 utilisée)")
    parser.add_argument("--max-samples", type=int, default=20000)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    densities: list[float] = []
    with Path(args.manifest).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            duration = float(entry["duration_s"])
            if duration < 1.0:
                continue  # trop court pour une mesure fiable
            n_tokens = len(tokenizer.encode(entry["text"].strip(), add_special_tokens=False))
            densities.append(n_tokens / duration)

    if not densities:
        raise SystemExit("aucun échantillon exploitable dans le manifest")

    densities.sort()
    median = statistics.median(densities)
    p90 = densities[int(0.9 * (len(densities) - 1))]

    # n_text tokens doivent couvrir l'audio des n_audio frames suivantes :
    # n_audio <= n_text * 12,5 / (densité * marge)
    n_audio = int(args.n_text * MIMI_FRAME_RATE / (p90 * args.margin))
    n_audio = max(1, n_audio)

    print(f"échantillons        : {len(densities)}")
    print(f"densité médiane     : {median:.2f} tokens/s")
    print(f"densité p90         : {p90:.2f} tokens/s (référence EN 6:12 → budget 6,25 t/s ; JP 6:9 → 8,33 t/s)")
    print(f"ratio suggéré       : {args.n_text}:{n_audio}  (marge {args.margin})")
    if n_audio > EN_RATIO[1]:
        print(f"→ densité FR plus faible que l'anglais : plafonner à {EN_RATIO[0]}:{EN_RATIO[1]} (défaut du modèle)")
    print("\nPasser le ratio retenu à preprocess_sft (--interleaved-text-tokens/--interleaved-audio-tokens).")
    print("⚠ Le ratio d'INFÉRENCE est figé dans la config du modèle (interleaved_n_text/n_audio) :")
    print("  l'ajuster aussi dans config.json du checkpoint exporté pour rester cohérent avec l'entraînement.")


if __name__ == "__main__":
    main()
