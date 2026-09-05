"""Generate brick C: French dialogues and code-switch dialogues.

Entry point: ``lfm2-generate-fr``. Text only — no GPU — so this runs while the
GPU work is queued.

    lfm2-generate-fr --out data/corpus/C_dialogues/dialogues.jsonl \\
      --n-fr 500 --n-switch 200

    lfm2-generate-fr --provider anthropic --model claude-sonnet-5 --batch \\
      --max-usd 2 --out corpus/C_dialogues/v2_parts/deep_01.jsonl --n-deep 300

Held-out contamination is checked against the evaluation benchmarks: a
generated utterance too close to a `lang_mirror` or `fr_s2s` case would train
on the very set that judges the result.

Spending: ``--max-usd`` stops the run cleanly (exit code 3) once the provider's
meter reaches the cap; everything produced so far is already on disk, and the
``===SPEND===`` line says what the run cost.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from lfm2_audio.cli.data.llm_providers import PROVIDERS, judge_stream, make_judge, spend_line
from lfm2_audio.data_prep.fr_dialogues import (
    TOPICS_EN,
    TOPICS_FR,
    FrDialogue,
    build_code_switch_prompt,
    build_deep_prompt,
    build_en_prompt,
    build_fr_prompt,
    build_social_prompt,
    code_switch_rate,
    parse_dialogues,
)
from lfm2_audio.data_prep.synth_dialogues import ContaminationFilter, _extract_json_array
from lfm2_audio.scorer.text.anthropic_judge import EFFORTS, parse_effort
from lfm2_audio.scorer.text.judge import Judge
from lfm2_audio.scorer.text.llm_spend import SpendCapReachedError

DEFAULT_BENCHMARKS = ("benchmark/lang_mirror/questions.jsonl", "benchmark/fr_s2s/questions.jsonl")
EXIT_SPEND_CAP = 3


@dataclass(frozen=True)
class Family:
    """One dialogue family: the prompts to send, in order, and how to label what comes back."""

    kind: str
    prefix: str
    target: int
    topics: list[str]
    prompts: list[str]


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


def parse_batch(raw: str, *, prefix: str, kind: str, start: int) -> list[FrDialogue]:
    try:
        payload = _extract_json_array(raw)
    except Exception as error:
        print(f"  lot ignoré ({error})", file=sys.stderr)
        return []
    return parse_dialogues(payload, prefix=prefix, kind=kind, start=start)


def _topics_per_call(target: int, per_call: int, topics: Sequence[str]) -> list[str]:
    return [topics[call % len(topics)] for call in range(len(range(0, target, per_call)))]


def _switch_family(target: int, per_call: int) -> Family:
    """Alternates the language order call by call, so both directions are learned."""
    labels: list[str] = []
    prompts: list[str] = []
    for call, topic in enumerate(_topics_per_call(target, per_call, TOPICS_FR)):
        first, second = ("français", "anglais") if call % 2 == 0 else ("anglais", "français")
        labels.append(f"{first}→{second}")
        prompts.append(build_code_switch_prompt(per_call, topic, first=first, second=second))
    return Family("code_switch", "c_cs", target, labels, prompts)


def plan_families(args: argparse.Namespace) -> list[Family]:
    """Every prompt of the run, decided up front: a batch judge sends a family as one lot."""
    builders: list[tuple[str, str, int, Sequence[str], Callable[[int, str], str]]] = [
        ("fr", "c_fr", args.n_fr, TOPICS_FR, build_fr_prompt),
        # The three families the coverage audit found missing: depth (627/695
        # dialogues had no history to learn), short social turns (v3's verdict),
        # and conversational English (the preservation share).
        ("fr_deep", "c_deep", args.n_deep, TOPICS_FR, build_deep_prompt),
        ("fr_social", "c_soc", args.n_social, TOPICS_FR, build_social_prompt),
        ("en", "c_en", args.n_en, TOPICS_EN, build_en_prompt),
    ]
    families: list[Family] = []
    for kind, prefix, target, topics, build in builders:
        if target > 0:
            picked = _topics_per_call(target, args.per_call, topics)
            families.append(Family(kind, prefix, target, picked, [build(args.per_call, t) for t in picked]))
    if args.n_switch > 0:
        families.insert(1 if families and families[0].kind == "fr" else 0, _switch_family(args.n_switch, args.per_call))
    return families


def run_family(
    judge: Judge,
    family: Family,
    filter_: ContaminationFilter,
    flush: Callable[[list[FrDialogue]], None],
) -> list[FrDialogue]:
    """Flushed EVERY batch. The per-family version lost an entire day: a 429
    mid-family discarded ~2000 in-memory dialogues whose API calls had already
    been paid for — twice, since the first run hung and its regeneration hit
    the spend cap."""
    produced: list[FrDialogue] = []
    for index, raw in enumerate(judge_stream(judge, family.prompts)):
        batch = parse_batch(raw, prefix=family.prefix, kind=family.kind, start=len(produced))
        produced += [d for d in batch if not filter_.is_contaminated(d.turns[0].text)]
        print(f"  {family.kind} {len(produced)}/{family.target} — {family.topics[index]}", flush=True)
        flush(produced)
    return produced


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-fr", type=int, default=500)
    parser.add_argument("--n-switch", type=int, default=200)
    parser.add_argument("--n-deep", type=int, default=0, help="dialogues longs à reprises anaphoriques (historique)")
    parser.add_argument("--n-social", type=int, default=0, help="micro-échanges sociaux (lacune du verdict v3)")
    parser.add_argument("--n-en", type=int, default=0, help="conversationnel anglais (part de préservation)")
    parser.add_argument("--per-call", type=int, default=10)
    parser.add_argument("--provider", choices=PROVIDERS, default="gemini")
    parser.add_argument("--model", default=None, help="défaut : selon --provider")
    parser.add_argument("--effort", choices=EFFORTS, default="low", help="Anthropic : profondeur de réflexion")
    parser.add_argument("--batch", action="store_true", help="Anthropic : API Message Batches, moitié prix")
    parser.add_argument(
        "--max-usd", type=float, default=None, help="plafond de dépense ; code de sortie 3 s'il est atteint"
    )
    parser.add_argument("--benchmarks", nargs="*", type=Path, default=[Path(p) for p in DEFAULT_BENCHMARKS])
    parser.add_argument("--contamination-threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        judge = make_judge(
            args.provider, args.model, max_usd=args.max_usd, batch=args.batch, effort=parse_effort(args.effort)
        )
    except ValueError as error:
        print(f"❌ {error}", file=sys.stderr)
        raise SystemExit(1) from error

    filter_ = ContaminationFilter(held_out=held_out_utterances(args.benchmarks), threshold=args.contamination_threshold)
    print(f"contamination : {len(filter_.held_out)} énoncés held-out")

    done: list[FrDialogue] = []
    by_kind: dict[str, list[FrDialogue]] = {}
    status = 0
    try:
        for family in plan_families(args):
            produced = run_family(judge, family, filter_, flush=_flusher(args.out, list(done)))
            done += produced
            by_kind[family.kind] = produced
    except SpendCapReachedError as error:
        # Everything produced so far was flushed by run_family; do NOT rewrite.
        print(
            f"⛔ {error} — arrêt propre, {len(done)} dialogues complets + le lot en cours sur disque", file=sys.stderr
        )
        status = EXIT_SPEND_CAP
    else:
        _write(args.out, done)

    switches = by_kind.get("code_switch", [])
    rate = code_switch_rate(switches) if switches else 0.0
    print(f"\nFR : {len(by_kind.get('fr', []))} · code-switch : {len(switches)} (dont {rate:.0%} changent de langue)")
    if switches and rate < 0.5:
        print("⚠️  moins d'un dialogue sur deux change réellement de langue — le lot n'entraînera pas le miroir")
    print(f"→ {args.out}")
    line = spend_line(judge)
    if line:
        print(line, flush=True)
    raise SystemExit(status)


def _flusher(out: Path, done: list[FrDialogue]) -> Callable[[list[FrDialogue]], None]:
    """Writes the finished families plus the family in progress, every batch."""
    return lambda partial: _write(out, done + partial)


def _write(out: Path, dialogues: list[FrDialogue]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for dialogue in dialogues:
            handle.write(json.dumps(dialogue.as_case(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
