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
import statistics
from pathlib import Path
from typing import Any

from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.text.hf_judge import HfJudge
from lfm2_audio.scorer.text.reasoning import ReasoningScorer


def to_sample(record: dict[str, Any]) -> EvalSample:
    """A transcript record → the sample shape every scorer consumes."""
    return EvalSample(
        sample_id=f"{record['scenario']}__t{record['turn']}",
        prompt_text=record["user"],
        predicted_text=record["answer"],
        tool_results=[
            {"name": tool.get("name"), "result": tool.get("result_preview", "")} for tool in record.get("tools", [])
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    judge = HfJudge()
    if not judge.has_credentials:
        raise SystemExit("HF_TOKEN absent: the judge cannot run")
    scorer = ReasoningScorer(judge)

    records = [json.loads(line) for line in args.transcript.open(encoding="utf-8")]
    rows = []
    for record in records:
        sample = to_sample(record)
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

    out = args.out or args.transcript.with_name(f"judged_{args.transcript.stem}.json")
    out.write_text(
        json.dumps({"rubric": scorer.rubric.version, "judge": judge.model_id, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
