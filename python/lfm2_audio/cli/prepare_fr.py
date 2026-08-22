"""Phase 2a — Construit les dialogues ASR/TTS FR (mode sequential) depuis un manifest.

Entrée : manifest JSONL ``{"audio": "rel/path.wav", "text": "transcription"}``
(exporté de Common Voice FR / MLS-FR / Emilia-FR — l'export lui-même relève de
la Phase 1). Sortie : JSONL au schéma ``dialogue_schema``, à packer ensuite avec
``python -m lfm2_audio.data_prep.preprocess_sft --assistant-audio-mode sequential``.

Deux gabarits, comme la recette du variant JP :
- ASR : user = audio  → assistant = texte (transcription) ;
- TTS : user = texte (consigne de lecture) → assistant = texte + audio.

Usage :
    lfm2-prepare-fr --manifest cv_fr.jsonl \\
        --output data/fr_asr_tts.jsonl --tasks asr tts --tts-ratio 0.5
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ASR_SYSTEM = "Transcris fidèlement l'audio en français."
ASR_USER_PROMPTS = [None]  # l'audio seul suffit en mode sequential
TTS_SYSTEM = "Lis le texte à voix haute en français, d'une voix naturelle."
TTS_USER_TEMPLATES = [
    "Lis ce texte : {text}",
    "Peux-tu lire à voix haute : {text}",
    "{text}",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSONL {audio, text}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", choices=["asr", "tts"], default=["asr", "tts"])
    parser.add_argument(
        "--tts-ratio", type=float, default=0.5, help="part des échantillons convertis en TTS (si les deux tâches)"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n_asr = n_tts = 0

    with Path(args.manifest).open(encoding="utf-8") as fin, Path(args.output).open("w", encoding="utf-8") as fout:
        for i, raw_line in enumerate(fin):
            if args.max_samples is not None and i >= args.max_samples:
                break
            line = raw_line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio, text = entry["audio"], entry["text"].strip()
            if not text:
                continue

            task = ("tts" if rng.random() < args.tts_ratio else "asr") if len(args.tasks) == 2 else args.tasks[0]

            if task == "asr":
                dialogue = {
                    "id": f"asr_{i:07d}",
                    "system": ASR_SYSTEM,
                    "turns": [
                        {"role": "user", "audio": audio},
                        {"role": "assistant", "text": text},
                    ],
                }
                n_asr += 1
            else:
                template = rng.choice(TTS_USER_TEMPLATES)
                dialogue = {
                    "id": f"tts_{i:07d}",
                    "system": TTS_SYSTEM,
                    "turns": [
                        {"role": "user", "text": template.format(text=text)},
                        {"role": "assistant", "text": text, "audio": audio},
                    ],
                }
                n_tts += 1

            fout.write(json.dumps(dialogue, ensure_ascii=False) + "\n")

    print(f"écrit {args.output} : {n_asr} ASR + {n_tts} TTS")
    print(
        "Packer ensuite avec : python -m lfm2_audio.data_prep.preprocess_sft "
        f"--dialogues {args.output} --audio-root <racine audio> --assistant-audio-mode sequential ..."
    )


if __name__ == "__main__":
    main()
