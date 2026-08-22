"""Réhydrate un dataset HF audio en JSONL + WAV (inverse de ``push_hf``).

Point d'entrée : ``lfm2-hf-to-dialogues``.
La mise en forme d'une ligne vit dans
:mod:`lfm2_audio.data_prep.hf_rehydrate` (Python pur, testée) ; le
téléchargement et l'écriture des WAV restent ici, avec leurs dépendances.

    lfm2-hf-to-dialogues --repo-id Rcarvalo/tc-en-audio-toolcalling --out data/tc_en
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

from lfm2_audio.data_prep.hf_rehydrate import row_to_dialogue


def rehydrate(repo_id: str, out: str | Path) -> dict[str, int]:
    """Télécharge le dataset et écrit, par split, ``<out>/<split>.jsonl`` + les WAV.

    Les WAV sont réécrits en PCM 16 bits à 16 kHz : c'est ce que l'encodeur
    attend, et le ré-encodage ici évite de le refaire à chaque entraînement.
    """
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split, raw_split in load_dataset(repo_id).items():
        rows = raw_split.cast_column("audio", Audio(sampling_rate=16_000))
        audio_root = destination / f"audio_{split}"
        audio_root.mkdir(parents=True, exist_ok=True)

        written = 0
        with (destination / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                relative = f"{row['id']}_u0.wav"
                sf.write(
                    str(audio_root / relative),
                    row["audio"]["array"],
                    row["audio"]["sampling_rate"],
                    subtype="PCM_16",
                )
                handle.write(json.dumps(row_to_dialogue(row, relative), ensure_ascii=False) + "\n")
                written += 1
        counts[split] = written

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out", default=Path("data/tc_en"), type=Path)
    args = parser.parse_args()

    for split, count in rehydrate(args.repo_id, args.out).items():
        print(f"{split}: {count} dialogues → {args.out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
