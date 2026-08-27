"""Generate brick C: French dialogues and code-switch dialogues.

Entry point: ``lfm2-generate-fr``. Text only — no GPU — so this runs while the
GPU work is queued.

    lfm2-generate-fr --out data/corpus/C_dialogues/dialogues.jsonl \\
      --n-fr 500 --n-switch 200

Held-out contamination is checked against the evaluation benchmarks: a
generated utterance too close to a `lang_mirror` or `fr_s2s` case would train
on the very set that judges the result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lfm2_audio.data_prep.fr_dialogues import (
    TOPICS_FR,
    FrDialogue,
    build_code_switch_prompt,
    build_fr_prompt,
    code_switch_rate,
    parse_dialogues,
)
from lfm2_audio.data_prep.synth_dialogues import ContaminationFilter, _extract_json_array
from lfm2_audio.scorer.text.gemini_judge import GeminiJudge

DEFAULT_BENCHMARKS = ("benchmark/lang_mirror/questions.jsonl", "benchmark/fr_s2s/questions.jsonl")


def held_out_utterances(paths: list[Path]) -> list[str]:
    utterances = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            utterances += [turn.get("text", "") for turn in case.get("turns", []) if turn.get("text")]
    return utterances


def generate(judge: GeminiJudge, prompt: str, *, prefix: str, kind: str, start: int) -> list[FrDialogue]:
    raw = judge.judge(prompt)
    try:
        payload = _extract_json_array(raw)
    except Exception as error:
        print(f"  lot ignoré ({error})", file=sys.stderr)
        return []
    return parse_dialogues(payload, prefix=prefix, kind=kind, start=start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-fr", type=int, default=500)
    parser.add_argument("--n-switch", type=int, default=200)
    parser.add_argument("--per-call", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--benchmarks", nargs="*", type=Path, default=[Path(p) for p in DEFAULT_BENCHMARKS])
    parser.add_argument("--contamination-threshold", type=float, default=0.5)
    args = parser.parse_args()

    judge = GeminiJudge(args.model) if args.model else GeminiJudge()
    if not judge.has_credentials:
        print("❌ GEMINI_API_KEY absent", file=sys.stderr)
        raise SystemExit(1)

    filter_ = ContaminationFilter(held_out=held_out_utterances(args.benchmarks), threshold=args.contamination_threshold)
    print(f"contamination : {len(filter_.held_out)} énoncés held-out")

    dialogues: list[FrDialogue] = []
    for index in range(0, args.n_fr, args.per_call):
        topic = TOPICS_FR[(index // args.per_call) % len(TOPICS_FR)]
        batch = generate(judge, build_fr_prompt(args.per_call, topic), prefix="c_fr", kind="fr", start=len(dialogues))
        dialogues += [d for d in batch if not filter_.is_contaminated(d.turns[0].text)]
        print(f"  FR {len(dialogues)}/{args.n_fr} — {topic}", flush=True)

    switches: list[FrDialogue] = []
    for index in range(0, args.n_switch, args.per_call):
        topic = TOPICS_FR[(index // args.per_call) % len(TOPICS_FR)]
        first, second = ("français", "anglais") if index % 2 == 0 else ("anglais", "français")
        batch = generate(
            judge,
            build_code_switch_prompt(args.per_call, topic, first=first, second=second),
            prefix="c_cs",
            kind="code_switch",
            start=len(switches),
        )
        switches += [d for d in batch if not filter_.is_contaminated(d.turns[0].text)]
        print(f"  code-switch {len(switches)}/{args.n_switch} — {first}→{second}", flush=True)

    rate = code_switch_rate(switches)
    print(f"\nFR : {len(dialogues)} · code-switch : {len(switches)} (dont {rate:.0%} changent vraiment de langue)")
    if switches and rate < 0.5:
        print("⚠️  moins d'un dialogue sur deux change réellement de langue — le lot n'entraînera pas le miroir")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for dialogue in dialogues + switches:
            handle.write(json.dumps(dialogue.as_case(), ensure_ascii=False) + "\n")
    print(f"→ {args.out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
