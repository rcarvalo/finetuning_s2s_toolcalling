"""Conversion d'un JSONL de dialogues en dataset Hugging Face.

La mise en forme des lignes et la carte du dataset vivent ici ; l'upload reste
dans la CLI ``lfm2-build-dataset``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_LICENSE = "cc-by-nc-4.0"


def dialogue_to_row(dlg: dict[str, Any], audio_root: str | Path) -> dict[str, Any]:
    """Dialogue single-turn → ligne plate (chemin audio + métadonnées + cible)."""
    user = next((t for t in dlg["turns"] if t.get("role") == "user" and t.get("audio")), None)
    assistant: dict[str, Any] = next((t for t in dlg["turns"] if t.get("role") == "assistant"), {})
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
    for raw_line in Path(jsonl).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            rows.append(dialogue_to_row(json.loads(line), audio_root))
    return rows


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

# {repo_id.rsplit("/", maxsplit=1)[-1]}

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
