"""Scoring BFCL-style des tool calls sur un JSONL de prédictions.

Point d'entrée : ``lfm2-eval-toolcalling``.
La logique vit dans :mod:`lfm2_audio.evaluation.toolcalling` — ce module ne porte que la CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lfm2_audio.evaluation.toolcalling import (
    evaluate_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSONL avec expected_calls + predicted_text")
    parser.add_argument("--output", default=None, help="JSON de sortie (défaut : stdout)")
    parser.add_argument("--per-case", action="store_true", help="Inclure le détail par cas")
    parser.add_argument(
        "--arg-match",
        choices=["exact", "token_f1", "semantic"],
        default="exact",
        help="comparaison des arguments string (texte libre db_query/web_search) ; token_f1/semantic = tolérant",
    )
    parser.add_argument(
        "--arg-threshold",
        type=float,
        default=0.7,
        help="seuil de similarité par argument (token_f1/semantic)",
    )
    args = parser.parse_args()

    report = evaluate_file(args.predictions, arg_match=args.arg_match, threshold=args.arg_threshold)
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
        print(
            "\n⚠ call_accuracy < 75% : en dessous du critère de validation Phase 2 "
            "(envisager le repli hybride audio-in/text-out + TTS)."
        )


if __name__ == "__main__":
    main()
