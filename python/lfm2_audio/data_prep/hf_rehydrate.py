"""Reconstruction d'un JSONL de dialogues depuis un dataset Hugging Face.

Opération inverse de :mod:`lfm2_audio.data_prep.hf_dataset`.
CLI : ``lfm2-hf-to-dialogues`` (:mod:`lfm2_audio.cli.data.pull_hf`).

``datasets`` et ``soundfile`` sont importés DANS :func:`rehydrate` : lire une
ligne ou lister les outils doit rester possible sans l'extra ``train``.
"""

from __future__ import annotations

import json
from typing import Any

TOOLS = ["web_search", "db_query"]


def row_to_dialogue(row: dict[str, Any], audio_rel: str) -> dict[str, Any]:
    """Ligne plate HF → dialogue single-turn (inverse de dialogue_to_row)."""
    user = {"role": "user", "text": row.get("utterance", ""), "audio": audio_rel}
    if row.get("voice"):
        user["voice"] = row["voice"]
    if row.get("has_tool_call"):
        assistant = {
            "role": "assistant",
            "tool_calls": [{"name": row["tool_name"], "arguments": json.loads(row["arguments"])}],
        }
    else:
        assistant = {"role": "assistant", "text": row.get("assistant_text") or ""}
    return {
        "id": row["id"],
        "tools": TOOLS,
        "meta": {k: row.get(k) for k in ("target", "style", "depth")},
        "turns": [user, assistant],
    }
