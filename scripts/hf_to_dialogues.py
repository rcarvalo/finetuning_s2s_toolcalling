#!/usr/bin/env python3
"""Réhydrate un dataset HF audio (push par build_hf_dataset) en JSONL + WAV.

Le dataset HF stocke des lignes plates (id, audio, utterance, tool_name, ...).
L'entraînement (``preprocess_sft`` + ``LFM2AudioChatMapper``) consomme lui un
JSONL au ``dialogue_schema`` + des fichiers WAV. Ce script télécharge le dataset
et écrit, par split, ``<out>/<split>.jsonl`` + ``<out>/audio_<split>/*.wav``.

    python scripts/hf_to_dialogues.py --repo-id Rcarvalo/tc-en-audio-toolcalling --out data/tc_en

``row_to_dialogue`` est l'inverse pur de ``build_hf_dataset.dialogue_to_row``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TOOLS = ["web_search", "db_query"]


def row_to_dialogue(row: dict[str, Any], audio_rel: str) -> dict[str, Any]:
    """Ligne plate HF → dialogue single-turn (inverse de dialogue_to_row)."""
    user = {"role": "user", "text": row.get("utterance", ""), "audio": audio_rel}
    if row.get("voice"):
        user["voice"] = row["voice"]
    if row.get("has_tool_call"):
        assistant = {"role": "assistant",
                     "tool_calls": [{"name": row["tool_name"], "arguments": json.loads(row["arguments"])}]}
    else:
        assistant = {"role": "assistant", "text": row.get("assistant_text") or ""}
    return {
        "id": row["id"],
        "tools": TOOLS,
        "meta": {k: row.get(k) for k in ("target", "style", "depth")},
        "turns": [user, assistant],
    }


def rehydrate(repo_id: str, out: str | Path) -> dict[str, int]:
    import soundfile as sf
    from datasets import Audio, load_dataset

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    dd = load_dataset(repo_id)
    for split, ds in dd.items():
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
        audio_root = out / f"audio_{split}"
        audio_root.mkdir(parents=True, exist_ok=True)
        n = 0
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in ds:
                rel = f"{row['id']}_u0.wav"
                sf.write(str(audio_root / rel), row["audio"]["array"], row["audio"]["sampling_rate"],
                         subtype="PCM_16")
                f.write(json.dumps(row_to_dialogue(row, rel), ensure_ascii=False) + "\n")
                n += 1
        counts[split] = n
        print(f"{split}: {n} dialogues -> {out / f'{split}.jsonl'} (+ {audio_root}/)")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out", default="data/tc_en", type=Path)
    args = ap.parse_args()
    rehydrate(args.repo_id, args.out)


if __name__ == "__main__":
    main()
