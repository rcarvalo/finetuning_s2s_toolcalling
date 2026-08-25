"""Judge the relevance of scenario answers with the LLM judge (no GPU needed).

The tool_call scorer measures whether the right tool was called; it cannot see
that an answer quotes the payload without answering the question — v3's Ballon
d'Or reply ("France Football modified the rules…") passed every mechanical
check. `ReasoningScorer` grades relevance and grounding separately, which is
exactly that distinction.

Runs on the recorded scenario transcript, so it re-scores past runs without
re-generating anything.

    python infra/judge_scenarios.py --transcript reports/scenarios_v3/transcript.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.text.hf_judge import HfJudge
from lfm2_audio.scorer.text.judge import Judge
from lfm2_audio.scorer.text.reasoning import ReasoningScorer


def to_sample(record: dict[str, Any]) -> EvalSample:
    """A transcript record → the sample shape every scorer consumes."""
    return EvalSample(
        sample_id=f"{record['scenario']}__t{record['turn']}",
        prompt_text=record["user"],
        predicted_text=record["answer"],
        # Prefer the full payload; `result_preview` (300 chars) only exists in
        # transcripts recorded before that fix, and grading grounding against a
        # truncated payload marks correct answers as hallucinations.
        tool_results=[
            {"name": tool.get("name"), "result": tool.get("result") or tool.get("result_preview", "")}
            for tool in record.get("tools", [])
        ],
    )


def build_judge() -> Judge:
    """Gemini when its key is present, Hugging Face otherwise.

    Gemini flash is the cheap path and the project's historical judge; the HF
    fallback exists so a missing key never silently turns `reasoning` into an
    UNAVAILABLE scorer.
    """
    if os.environ.get("GEMINI_API_KEY"):
        from lfm2_audio.scorer.text.gemini_judge import GeminiJudge

        return GeminiJudge()
    judge = HfJudge()
    if not judge.has_credentials:
        raise SystemExit("ni GEMINI_API_KEY ni HF_TOKEN : aucun juge disponible")
    return judge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="hard cap on judged turns. Every judged turn is a paid API call, so "
        "the cap is on by default and must be raised deliberately.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-judge turns already present in the output file (default: reuse them)",
    )
    args = parser.parse_args()

    judge = build_judge()
    scorer = ReasoningScorer(judge)

    out = args.out or args.transcript.with_name(f"judged_{args.transcript.stem}.json")
    cached: dict[str, dict[str, Any]] = {}
    if out.exists() and not args.force:
        previous = json.loads(out.read_text(encoding="utf-8"))
        cached = {row["id"]: row for row in previous.get("rows", []) if row.get("status") == "OK"}
        if cached:
            print(f"{len(cached)} tours déjà jugés, réutilisés (--force pour refaire)", flush=True)

    records = [json.loads(line) for line in args.transcript.open(encoding="utf-8")]
    rows = []
    spent = 0
    for record in records:
        sample = to_sample(record)
        if sample.sample_id in cached:
            rows.append(cached[sample.sample_id])
            continue
        if spent >= args.limit:
            print(f"plafond de {args.limit} appels atteint, {len(records) - len(rows)} tours non jugés", flush=True)
            break
        spent += 1
        result = scorer.score(sample)
        detail = result.details or {}
        rows.append(
            {
                "id": sample.sample_id,
                "tooled": bool(record.get("tools")),
                "status": result.status.name,
                "value": result.value,
                # ReasoningScorer flattens the criteria into details alongside
                # rationale/rubric_version — there is no nested "scores" key.
                "scores": {k: v for k, v in detail.items() if k not in ("rationale", "rubric_version")},
                "rationale": detail.get("rationale", ""),
                "question": record["user"],
                "answer": record["answer"][:160],
            }
        )
        print(f"{rows[-1]['id']:34s} {rows[-1]['status']:11s} {rows[-1]['scores']}", flush=True)

    scored = [r for r in rows if r["value"] is not None]
    if scored:
        print(f"\nnoted {len(scored)}/{len(rows)} — mean {statistics.mean(r['value'] for r in scored):.3f}")
        for key in ("relevance", "grounding", "coherence", "conciseness"):
            values = [r["scores"][key] for r in scored if key in r["scores"]]
            if values:
                print(f"  {key:12s} {statistics.mean(values):.2f}")

    out.write_text(
        json.dumps({"rubric": scorer.rubric.version, "judge": judge.model_id, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
