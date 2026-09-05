"""Évaluation BFCL-style du tool calling sur set FR maison (Phases 2 & 5).

Scoring AST (pas d'exécution) : on compare les appels extraits du texte généré
aux appels attendus. Catégories rapportées :

- ``call_accuracy``      : nom + arguments exacts (match non ordonné des appels) ;
- ``name_accuracy``      : nom de fonction correct (arguments ignorés) ;
- ``relevance_accuracy`` : décision appeler / ne pas appeler correcte
                           (les négatifs « irrelevance » comptent ici) ;
- ``parse_rate``         : proportion de sorties dont le span d'appel est parsable.

Format du JSONL d'éval (une ligne par cas) :

    {"id": "case_001",
     "expected_calls": [{"name": "check_appointment",
                         "arguments": {"visitor_name": "Marie Dupont"}}],
     "predicted_text": "<|tool_call_start|>[check_appointment(visitor_name=\\"Marie Dupont\\")]<|tool_call_end|>"}

``expected_calls: []`` pour les cas négatifs. ``predicted_text`` est le flux
texte brut du modèle (l'inférence elle-même est hors scope de ce module : voir
``orchestrator.agent`` ou un harnais batch dédié).

Usage :
    python -m lfm2_audio.evaluation.toolcalling --predictions eval_fr.jsonl
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lfm2_audio.evaluation.argument_match import ArgMatch, diff_arguments, token_f1
from lfm2_audio.evaluation.tool_call_diagnosis import diagnose

# `token_f1` and `ArgMatch` moved to `argument_match` when the diagnosis needed
# them without importing this module; re-exported because this module is the
# public façade callers and tests already import.
__all__ = [
    "ArgMatch",
    "CaseResult",
    "Report",
    "calls_match",
    "evaluate_file",
    "score_case",
    "token_f1",
]


def calls_match(
    predicted: dict[str, Any],
    expected: dict[str, Any],
    *,
    arg_match: ArgMatch = "exact",
    threshold: float = 0.7,
) -> bool:
    if predicted["name"] != expected["name"]:
        return False
    return not diff_arguments(
        predicted.get("arguments", {}),
        expected.get("arguments", {}),
        arg_match=arg_match,
        threshold=threshold,
    )


@dataclass(slots=True)
class CaseResult:
    case_id: str
    parsed: bool
    expected_call: bool
    predicted_call: bool
    name_correct: bool
    call_correct: bool


@dataclass
class Report:
    results: list[CaseResult] = field(default_factory=list)

    def add(self, r: CaseResult) -> None:
        self.results.append(r)

    def summary(self) -> dict[str, Any]:
        n = len(self.results)
        if n == 0:
            return {"cases": 0}
        positives = [r for r in self.results if r.expected_call]
        return {
            "cases": n,
            "positives": len(positives),
            "negatives": n - len(positives),
            "parse_rate": sum(r.parsed for r in self.results) / n,
            "relevance_accuracy": sum(r.expected_call == r.predicted_call for r in self.results) / n,
            "name_accuracy": (sum(r.name_correct for r in positives) / len(positives)) if positives else None,
            "call_accuracy": (sum(r.call_correct for r in positives) / len(positives)) if positives else None,
        }


def score_case(
    case_id: str,
    predicted_text: str,
    expected_calls: list[dict[str, Any]],
    *,
    arg_match: ArgMatch = "exact",
    threshold: float = 0.7,
) -> CaseResult:
    """Verdicts only. ``diagnose`` is the same computation with the evidence kept."""
    diagnosis = diagnose(case_id, predicted_text, expected_calls, arg_match=arg_match, threshold=threshold)
    return CaseResult(
        case_id=diagnosis.case_id,
        parsed=diagnosis.parsed,
        expected_call=diagnosis.expected_call,
        predicted_call=diagnosis.predicted_call,
        name_correct=diagnosis.name_correct,
        call_correct=diagnosis.call_correct,
    )


def evaluate_file(path: str | Path, *, arg_match: ArgMatch = "exact", threshold: float = 0.7) -> Report:
    report = Report()
    with Path(path).open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            case = json.loads(line)
            report.add(
                score_case(
                    str(case["id"]),
                    case["predicted_text"],
                    case.get("expected_calls", []),
                    arg_match=arg_match,
                    threshold=threshold,
                )
            )
    return report
