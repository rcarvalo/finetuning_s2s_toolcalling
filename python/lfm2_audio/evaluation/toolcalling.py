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
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from lfm2_audio.orchestrator.tool_parser import StreamingToolCallParser


def _norm_value(v: Any) -> Any:  # noqa: ANN401 — valeur JSON d'argument
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


ArgMatch = str  # "exact" | "token_f1" | "semantic"


def _token_set(s: str) -> set[str]:
    norm = _norm_value(s)
    return set(norm.split()) if isinstance(norm, str) else set()


def token_f1(a: str, b: str) -> float:
    """F1 symétrique des tokens normalisés — tolérant à l'ordre/paraphrase légère."""
    ta, tb = _token_set(a), _token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    precision, recall = inter / len(tb), inter / len(ta)
    return 2 * precision * recall / (precision + recall)


_EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embedder() -> Any:  # noqa: ANN401 — SentenceTransformer non typé
    """Modèle d'embeddings, chargé au premier usage et gardé en mémoire.

    ``lru_cache`` plutôt qu'un global mutable : même effet (un seul chargement),
    sans état modifiable au niveau module.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_EMBEDDER_NAME)


def _semantic_sim(a: str, b: str) -> float:
    """Similarité cosinus d'embeddings (sentence-transformers, paresseux)."""
    emb = _embedder().encode([a, b], normalize_embeddings=True)
    return float(emb[0] @ emb[1])


def _args_match(pred_args: dict[str, Any], exp_args: dict[str, Any], *, arg_match: ArgMatch, threshold: float) -> bool:
    if set(pred_args) != set(exp_args):
        return False
    for key, exp_v in exp_args.items():
        pred_v = pred_args[key]
        if arg_match != "exact" and isinstance(pred_v, str) and isinstance(exp_v, str):
            sim = token_f1(pred_v, exp_v) if arg_match == "token_f1" else _semantic_sim(pred_v, exp_v)
            if sim < threshold:
                return False
        elif _norm_value(pred_v) != _norm_value(exp_v):
            return False
    return True


def calls_match(
    predicted: dict[str, Any],
    expected: dict[str, Any],
    *,
    arg_match: ArgMatch = "exact",
    threshold: float = 0.7,
) -> bool:
    if predicted["name"] != expected["name"]:
        return False
    return _args_match(
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
    parser = StreamingToolCallParser()
    predicted = [{"name": c.name, "arguments": c.arguments} for c in parser.feed(predicted_text)]
    parse_failed = bool(parser.errors)

    expected_call = bool(expected_calls)
    predicted_call = bool(predicted) or parse_failed  # une tentative malformée reste une tentative d'appel

    name_correct = False
    call_correct = False
    if expected_call and predicted:
        exp_names = sorted(str(c["name"]) for c in expected_calls)
        pred_names = sorted(str(c["name"]) for c in predicted)
        name_correct = exp_names == pred_names

        remaining = list(expected_calls)
        matched = 0
        for p in predicted:
            for e in remaining:
                if calls_match(p, e, arg_match=arg_match, threshold=threshold):
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
