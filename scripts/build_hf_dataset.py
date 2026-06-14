#!/usr/bin/env python3
"""Assemble les dialogues TTS en dataset HF (Audio + colonnes) et push sur le Hub.

Entrée : les JSONL de dialogues APRÈS TTS (train + test), au ``dialogue_schema``
single-turn (tour user avec ``audio``, tour assistant tool_calls/text). Sortie :
un ``DatasetDict`` {train, test} avec la feature ``Audio`` 16 kHz, poussé sur le
Hub (privé par défaut). La carte du dataset reste neutre (synthèse + TTS), sans
nommer le moteur de synthèse.

    python scripts/build_hf_dataset.py --repo-id Rcarvalo/tc-en-audio \
        --train data/tc_en_train.audio.jsonl --test data/tc_en_bench.audio.jsonl \
        --audio-root data/audio_tc_en --private

NB licence : l'audio est synthétisé par un modèle TTS sous CC-BY-NC-4.0 → le
dataset est marqué non-commercial (recherche). Garde le repo privé si tu n'es
pas sûr de tes droits de redistribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_LICENSE = "cc-by-nc-4.0"


def dialogue_to_row(dlg: dict[str, Any], audio_root: str | Path) -> dict[str, Any]:
    """Dialogue single-turn → ligne plate (chemin audio + métadonnées + cible)."""
    user = next((t for t in dlg["turns"] if t.get("role") == "user" and t.get("audio")), None)
    assistant = next((t for t in dlg["turns"] if t.get("role") == "assistant"), {})
    if user is None:
        raise ValueError(f"{dlg.get('id')}: aucun tour user avec audio (TTS manquant ?)")

    tool_calls = assistant.get("tool_calls", [])
    expected = [{"name": c["name"], "arguments": c.get("arguments", {})} for c in tool_calls]
    meta = dlg.get("meta", {})
    return {
        "id": dlg["id"],
        "audio": str(Path(audio_root) / user["audio"]),
        "utterance": user.get("text", ""),
        "has_tool_call": bool(tool_calls),
        "tool_name": expected[0]["name"] if expected else None,
        "arguments": json.dumps(expected[0]["arguments"]) if expected else None,
        "assistant_text": assistant.get("text"),
        "expected_calls": json.dumps(expected),
        "voice": user.get("voice"),
        "target": meta.get("target"),
        "style": meta.get("style"),
        "depth": meta.get("depth"),
    }


def load_rows(jsonl: str | Path, audio_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(jsonl).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(dialogue_to_row(json.loads(line), audio_root))
    return rows


def _build_split(jsonl: str | Path, audio_root: str | Path):
    from datasets import Audio, Dataset

    ds = Dataset.from_list(load_rows(jsonl, audio_root))
    return ds.cast_column("audio", Audio(sampling_rate=16_000))


def dataset_card(repo_id: str, license_id: str, n_train: int, n_test: int) -> str:
    return f"""---
license: {license_id}
task_categories:
- audio-classification
- automatic-speech-recognition
language:
- en
tags:
- tool-calling
- function-calling
- speech
- voice-agent
---

# {repo_id.split('/')[-1]}

Synthetic **English** spoken tool-calling dataset for a voice assistant with two
tools — `web_search` and `db_query` (natural-language question). Each example is
a single spoken user utterance paired with the correct action: a tool call or no
call (negative). Audio is synthesized speech (multiple voices); the `test` split
uses voices held out from `train`.

- **train**: {n_train} examples · **test**: {n_test} examples (held-out voices)
- Tool-call format: pythonic, e.g. `web_search(query="...")`, `db_query(question="...")`.

## Columns
- `audio`: 16 kHz mono speech of the user utterance.
- `utterance`: the spoken text.
- `has_tool_call`: whether a tool should be called.
- `tool_name`, `arguments` (JSON): the expected call (null for negatives).
- `assistant_text`: spoken reply for negatives.
- `expected_calls` (JSON): list form, for BFCL-style scoring.
- `voice`, `target`, `style`, `depth`: generation metadata.

## Intended use
Research / non-commercial fine-tuning of speech tool-calling models. The audio is
machine-generated.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="ex. Rcarvalo/tc-en-audio")
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test", type=Path, default=None)
    ap.add_argument("--audio-root", required=True, type=Path)
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--license", default=DEFAULT_LICENSE)
    args = ap.parse_args()

    from datasets import DatasetDict
    from huggingface_hub import HfApi

    splits = {"train": _build_split(args.train, args.audio_root)}
    if args.test:
        splits["test"] = _build_split(args.test, args.audio_root)
    dd = DatasetDict(splits)
    print({k: v.num_rows for k, v in dd.items()})

    dd.push_to_hub(args.repo_id, private=args.private)

    card = dataset_card(args.repo_id, args.license, dd["train"].num_rows,
                        dd.get("test").num_rows if "test" in dd else 0)
    HfApi().upload_file(
        path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
        repo_id=args.repo_id, repo_type="dataset",
    )
    print(f"poussé sur https://huggingface.co/datasets/{args.repo_id} (private={args.private})")


if __name__ == "__main__":
    main()
