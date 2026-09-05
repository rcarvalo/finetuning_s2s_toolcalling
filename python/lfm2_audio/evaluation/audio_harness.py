"""Harnais d'évaluation du tool calling à partir d'audio.

Prédécesseur de :mod:`lfm2_audio.evaluation.pipeline`, conservé parce qu'il
alimente encore ``lfm2-eval-audio`` et ses tests. La fonction de prédiction est
injectée : le harnais se teste donc sans modèle.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from avet.scorers.toolcall.argument_match import ArgMatch

from lfm2_audio.evaluation.toolcalling import Report, score_case

PredictFn = Callable[[dict[str, Any]], str]  # case -> texte généré (avec le span d'appel)


def _to_eval_case(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalise une ligne en ``{id, audio, expected_calls}``.

    Accepte soit ce format direct, soit un dialogue single-turn au
    ``dialogue_schema`` (user audio → assistant tool_calls/text) — ainsi un
    même fichier sert au TTS ET à l'éval.
    """
    if "turns" not in obj:
        return obj
    audio = next((t.get("audio") for t in obj["turns"] if t.get("role") == "user"), None)
    expected: list[dict[str, Any]] = []
    for t in obj["turns"]:
        if t.get("role") == "assistant":
            expected = [{"name": c["name"], "arguments": c.get("arguments", {})} for c in t.get("tool_calls", [])]
    return {"id": obj["id"], "audio": audio, "expected_calls": expected}


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            cases.append(_to_eval_case(json.loads(line)))
    return cases


def run_audio_eval(
    cases: list[dict[str, Any]],
    predict_fn: PredictFn,
    *,
    arg_match: ArgMatch = "exact",
    threshold: float = 0.7,
) -> tuple[list[dict[str, Any]], Report]:
    """Exécute ``predict_fn`` sur chaque cas et score. Retourne (prédictions, rapport)."""
    predictions: list[dict[str, Any]] = []
    report = Report()
    for case in cases:
        predicted_text = predict_fn(case)
        expected = case.get("expected_calls", [])
        predictions.append({"id": case["id"], "expected_calls": expected, "predicted_text": predicted_text})
        report.add(score_case(str(case["id"]), predicted_text, expected, arg_match=arg_match, threshold=threshold))
    return predictions, report
