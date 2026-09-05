"""Refuse a bad Anthropic key, model or effort before spending a session on it.

Same lesson as the Gemini preflight: the generator only finds out at its first
real call, after the whole bootstrap. One tiny request through the SAME request
shape the run will use (model, effort, streaming) says so in one line, and
prints the key's shape (never the key) for comparison with the one that works.
"""

from __future__ import annotations

import os
import sys

from lfm2_audio.scorer.text.anthropic_judge import AnthropicJudge, parse_effort


def preflight(model: str, effort: str = "low") -> None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    shape = f"préfixe {key[:7]!r}, {len(key)} caractères"
    if not key.strip():
        sys.exit("ANTHROPIC_API_KEY manquant ou vide — ligne ANTHROPIC_API_KEY= du .env (ou secret Colab)")
    judge = AnthropicJudge(model, api_key=key, effort=parse_effort(effort), max_tokens=8)
    try:
        judge.judge("Reply with: ok")
    except Exception as error:  # the SDK raises several classes; any of them means "not usable"
        sys.exit(
            f"Anthropic refuse la requête ({type(error).__name__}: {str(error)[:140]}). "
            f"Modèle {model}, effort {effort}, clé : {shape}."
        )
    print(f"Anthropic OK ({model}, effort {effort}, {shape})", flush=True)
