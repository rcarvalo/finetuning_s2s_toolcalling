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
    python -m s2s_toolcalling.evaluation.eval_toolcalling --predictions eval_fr.jsonl
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from s2s_toolcalling.orchestrator.tool_parser import StreamingToolCallParser


def _norm_value(v: Any) -> Any:
    """Normalisation tolérante : accents/casse/espaces pour les strings, récursif sinon."""
    if isinstance(v, str):
        s = "".join(c for c in unicodedata.normalize("NFD", v.lower().strip()) if unicodedata.category(c) != "Mn")
        return " ".join(s.split())
    if isinstance(v, dict):
        return {k: _norm_value(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [_norm_value(x) for x in v]
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def calls_match(predicted: dict[str, Any], expected: dict[str, Any]) -> bool:
    if predicted["name"] != expected["name"]:
        return False
    pred_args = {k: _norm_value(v) for k, v in predicted.get("arguments", {}).items()}
    exp_args = {k: _norm_value(v) for k, v in expected.get("arguments", {}).items()}
    return pred_args == exp_args


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


def score_case(case_id: str, predicted_text: str, expected_calls: list[dict[str, Any]]) -> CaseResult:
    parser = StreamingToolCallParser()
    predicted = [{"name": c.name, "arguments": c.arguments} for c in parser.feed(predicted_text)]
    parse_failed = bool(parser.errors)

    expected_call = bool(expected_calls)
    predicted_call = bool(predicted) or parse_failed  # une tentative malformée reste une tentative d'appel

    name_correct = False
    call_correct = False
    if expected_call and predicted:
        exp_names = sorted(c["name"] for c in expected_calls)
        pred_names = sorted(c["name"] for c in predicted)
        name_correct = exp_names == pred_names

        remaining = list(expected_calls)
        matched = 0
        for p in predicted:
            for e in remaining:
                if calls_match(p, e):
                    remaining.remove(e)
                    matched += 1
                    break
        call_correct = matched == len(expected_calls) == len(predicted)

    return CaseResult(
        case_id=case_id,
        parsed=not parse_failed,
        expected_call=expected_call,
        predicted_call=predicted_call,
        name_correct=name_correct,
        call_correct=call_correct,
    )


def evaluate_file(path: str | Path) -> Report:
    report = Report()
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            report.add(score_case(str(case["id"]), case["predicted_text"], case.get("expected_calls", [])))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSONL avec expected_calls + predicted_text")
    parser.add_argument("--output", default=None, help="JSON de sortie (défaut : stdout)")
    parser.add_argument("--per-case", action="store_true", help="Inclure le détail par cas")
    args = parser.parse_args()

    report = evaluate_file(args.predictions)
    out: dict[str, Any] = {"summary": report.summary()}
    if args.per_case:
        out["cases"] = [vars(r) for r in report.results]

    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)

    # Rappel des seuils de la méthodologie (Phase 2 : BFCL-style FR > 75 %).
    summary = report.summary()
    if summary.get("call_accuracy") is not None and summary["call_accuracy"] < 0.75:
        print("\n⚠ call_accuracy < 75% : en dessous du critère de validation Phase 2 "
              "(envisager le repli hybride audio-in/text-out + TTS).")


if __name__ == "__main__":
    main()
