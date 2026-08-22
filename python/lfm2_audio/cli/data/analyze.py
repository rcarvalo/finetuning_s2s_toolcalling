#!/usr/bin/env python3
"""QA d'un dataset tool-calling (JSONL ``dialogue_schema``, avant ou après TTS).

Répond à « ai-je un bon dataset ? » avec des chiffres et des **drapeaux** (seuils) :
distribution outils/négatifs, doublons, qualité des arguments, artefacts à trous,
et — si l'audio est présent — durées / silence / fichiers manquants / voix.

    lfm2-analyze-dataset --dialogues data/tc_en_train.audio.jsonl \
        --audio-root data/audio_tc_en

Les fonctions de stats sont pures (testables sans audio) ; ``audio_quality`` lit
juste les en-têtes WAV (+ RMS sur un échantillon).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lfm2_audio.data_prep.dataset_report import (
    analyze,
    flags,
    load_rows,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialogues", required=True, type=Path)
    ap.add_argument("--audio-root", type=Path, default=None)
    args = ap.parse_args()

    rep = analyze(load_rows(args.dialogues), args.audio_root)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    issues = flags(rep)
    print("\n" + ("✅ dataset sain — aucun drapeau" if not issues else "⚠️ drapeaux :\n - " + "\n - ".join(issues)))


if __name__ == "__main__":
    main()
