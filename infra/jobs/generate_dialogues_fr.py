#!/usr/bin/env python3
"""Generate the dialogue families the corpus audit found missing, shard by shard, to the Hub.

627 of the 695 dialogues in `C_dialogues` have exactly two user turns — there is
almost no history for the model to learn. `lfm2-generate-fr` already knows how
to write what is missing (`--n-deep`, `--n-social`, `--n-en`, `--n-switch`);
it had simply never been asked. Text only, CPU runtime.

    LFM2_JOB=generate_dialogues_fr LFM2_ARGS="--deep 3000 --social 3000 --en 1200 --switch 800"
    LFM2_ARGS="--provider anthropic --model claude-sonnet-5 --max-usd 7 --deep 3000 ..."

Why shards: the generator flushes every batch to its output file but cannot
resume from it — a lost VM would start the family over, paying every call
again. Each shard here is a separate generator run whose file goes to the Hub
the moment it exists (`C_dialogues/v2_parts/`), and a shard already on the Hub
is downloaded rather than regenerated. Ids are re-stamped with the shard index
at merge time, because every generator run numbers from zero.

Spending (Anthropic): `--max-usd` is the cap for the WHOLE job. Each shard
runs under its share of what is left, reports `===SPEND===`, and the ledger
decides whether the next one starts. A shard cut by its cap is still pushed:
it was paid for. `===PROJECTION===` says what finishing everything would cost.

The merged file lands at `C_dialogues/dialogues_v2.jsonl` — the path the
assistant-voice notebook already lists as a future source, so voicing it is
one line in `BRICK_A_SOURCES`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

JOBS_DIR = Path(__file__).resolve().parent
if str(JOBS_DIR) not in sys.path:
    sys.path.insert(0, str(JOBS_DIR))
from _budget import Budget  # noqa: E402
from _llm_run import LlmArgs, run_capturing  # noqa: E402

ROOT = Path(os.environ.get("LFM2_ROOT", "/content/finetuning_s2s_toolcalling"))
CORPUS_REPO = os.environ.get("LFM2_CORPUS_REPO", "Rcarvalo/lfm25-fr-corpus-v1")
PARTS_DIR = Path("C_dialogues/v2_parts")
MERGED = Path("C_dialogues/dialogues_v2.jsonl")
FAMILIES = {"deep": "--n-deep", "social": "--n-social", "en": "--n-en", "switch": "--n-switch"}
"""Family -> the generator flag that requests it; every other family is forced to 0."""
EXIT_SPEND_CAP = 3
DEFAULT_MODELS = {"gemini": None, "anthropic": "claude-opus-5"}


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def reid(rows: list[dict[str, Any]], family: str, shard: int) -> list[dict[str, Any]]:
    """Stamp the shard into every id: each generator run numbers from zero.

    `c_deep_0012` from shard 3 becomes `c_deep_s03_0012`; two shards can then
    never collide, and the original ordinal stays readable.
    """
    stamped = []
    for row in rows:
        original = str(row.get("id", ""))
        prefix, _, ordinal = original.rpartition("_")
        stamped_row = dict(row)
        stamped_row["id"] = f"{prefix or family}_s{shard:02d}_{ordinal or '0000'}"
        stamped.append(stamped_row)
    return stamped


def depth_distribution(rows: list[dict[str, Any]]) -> dict[int, int]:
    """User turns per dialogue — the number this whole job exists to move."""
    counter: collections.Counter[int] = collections.Counter()
    for row in rows:
        counter[sum(1 for t in row.get("turns", []) if t.get("role") == "user")] += 1
    return dict(sorted(counter.items()))


def hub_files() -> set[str]:
    from huggingface_hub import HfApi

    return set(HfApi().list_repo_files(CORPUS_REPO, repo_type="dataset"))


def push(relative: Path) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(ROOT / "corpus" / relative),
        path_in_repo=str(relative),
        repo_id=CORPUS_REPO,
        repo_type="dataset",
    )
    print(f"pushed {relative} -> {CORPUS_REPO}", flush=True)


def fetch(relative: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local = ROOT / "corpus" / relative
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(Path(hf_hub_download(CORPUS_REPO, str(relative), repo_type="dataset")).read_bytes())
    return local


def shard_command(family: str, size: int, per_call: int, out: Path, llm: LlmArgs, max_usd: float | None) -> list[str]:
    cmd = [sys.executable, "-m", "lfm2_audio.cli.data.generate_fr", "--out", str(out), "--n-fr", "0"]
    cmd += ["--per-call", str(per_call)]
    for flag in FAMILIES.values():
        cmd += [flag, str(size if flag == FAMILIES[family] else 0)]
    return cmd + llm.flags(max_usd)


def generate_shard(
    family: str, shard: int, size: int, per_call: int, llm: LlmArgs, budget: Budget, parallel: int
) -> Path | None:
    """One generator run; None when the cap stopped it before anything came back."""
    relative = PARTS_DIR / f"{family}_{shard:02d}.jsonl"
    local = ROOT / "corpus" / relative
    result = run_capturing(shard_command(family, size, per_call, local, llm, budget.allowance(parallel)), ROOT)
    if result.usd is not None:
        budget.record(result.usd, len(read_rows(local)))
    if result.status == EXIT_SPEND_CAP:
        budget.exhaust()
        print(f"⛔ plafond atteint pendant {relative} — ce qui existe est poussé, rien d'autre ne démarre", flush=True)
    elif result.status != 0:
        raise SystemExit(f"shard {relative}: générateur en échec (code {result.status})")
    if not read_rows(local):
        if result.status == EXIT_SPEND_CAP:
            return None
        raise SystemExit(f"shard {relative} came back empty — quota? rerun later, nothing is lost")
    push(relative)
    return local


def run_family(
    family: str, target: int, shard_size: int, per_call: int, llm: LlmArgs, budget: Budget, parallel: int
) -> list[dict[str, Any]]:
    """Every shard of one family, in order, re-stamped; the Hub decides what is left to do."""
    rows: list[dict[str, Any]] = []
    for shard in range((target + shard_size - 1) // shard_size):
        size = min(shard_size, target - shard * shard_size)
        relative = PARTS_DIR / f"{family}_{shard:02d}.jsonl"
        if str(relative) in hub_files():
            local = fetch(relative)
            print(f"shard {relative}: already on the Hub, reused", flush=True)
        elif not budget.can_start():
            print(f"budget épuisé : shard {relative} non lancé", flush=True)
            break
        else:
            maybe = generate_shard(family, shard, size, per_call, llm, budget, parallel)
            if maybe is None:
                break
            local = maybe
        produced = read_rows(local)
        rows += reid(produced, family, shard)
        print(f"===RESULT=== shard family={family} n={shard} rows={len(produced)}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", type=int, default=3000, help="long dialogues with anaphoric callbacks")
    parser.add_argument("--social", type=int, default=3000, help="short social exchanges (v3's weakest register)")
    parser.add_argument("--en", type=int, default=1200, help="conversational English, the preservation share")
    parser.add_argument("--switch", type=int, default=800, help="code-switch dialogues on top of the 196 that exist")
    parser.add_argument("--shard-size", type=int, default=300)
    parser.add_argument("--per-call", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=4, help="families generated side by side (1 = sequential)")
    parser.add_argument("--provider", choices=("gemini", "anthropic"), default="gemini")
    parser.add_argument("--model", default=None, help="défaut : selon --provider")
    parser.add_argument("--effort", default="low", help="Anthropic : profondeur de réflexion")
    parser.add_argument(
        "--batch", action=argparse.BooleanOptionalAction, default=None, help="Anthropic : Batches (défaut oui)"
    )
    parser.add_argument(
        "--max-usd", type=float, default=None, help="plafond pour tout le job (obligatoire avec Anthropic)"
    )
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> LlmArgs:
    """Refuse a bad key, model or missing cap before any shard starts."""
    model = args.model or DEFAULT_MODELS[args.provider]
    if args.provider == "anthropic":
        from _anthropic_preflight import preflight as check

        if args.max_usd is None:
            sys.exit("--max-usd obligatoire avec Anthropic : le budget du projet est de 10 € tous services confondus")
        check(str(model), args.effort)
    else:
        from _gemini_preflight import preflight as check_gemini

        check_gemini()
    batch = args.batch if args.batch is not None else args.provider == "anthropic"
    return LlmArgs(args.provider, model, args.effort, batch)


def main() -> None:
    args = parse_args()
    llm = preflight(args)
    budget = Budget(args.max_usd)
    targets = {"deep": args.deep, "social": args.social, "en": args.en, "switch": args.switch}
    # One thread per family: the generator is sequential and a deep dialogue is
    # a long completion, so four families in a row measured 7-12 h; side by
    # side they share the wall clock. Shards stay independent files, so the
    # Hub is asked before EACH shard — a shard pushed by another run, or by an
    # earlier attempt, is reused rather than regenerated.
    parallel = max(1, args.parallel)
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        per_family = list(
            pool.map(
                lambda item: run_family(item[0], item[1], args.shard_size, args.per_call, llm, budget, parallel),
                targets.items(),
            )
        )
    merged: list[dict[str, Any]] = [row for rows in per_family for row in rows]

    write_rows(ROOT / "corpus" / MERGED, merged)
    push(MERGED)
    kinds = collections.Counter((row.get("meta") or {}).get("kind", "?") for row in merged)
    depth = depth_distribution(merged)
    print(
        f"===RESULT=== merged={MERGED} dialogues={len(merged)} by_kind={dict(kinds)} user_turns={depth}",
        flush=True,
    )
    print(budget.summary(missing=max(0, sum(targets.values()) - len(merged))), flush=True)


if __name__ == "__main__":
    main()
