"""Is the French TEXT broken, or only broken while audio is being interleaved?

The 0B baseline found two French failures that may be one. The model's French
text injects English mid-sentence ("La règle du hors-healthcare", "En été à
Marseille technological services"), and its French speech runs ~1.7x too long
per character before degenerating. Both would follow from an interleaving
schedule calibrated on English: 6 text tokens then 12 audio frames, forced on a
language whose tokens buy less speech.

So: same questions, same model, two decoding modes.

  text_only    no audio interleaved at all (GenerationConfig.text_only)
  interleaved  the normal path

Clean French in text-only and contaminated French interleaved would mean the
schedule corrupts generation, and R2 becomes a calibration problem rather than
a data problem — two opposite plans, decided by one hour of GPU.

The job only GENERATES and saves both texts; the comparison is judged off-pod,
where the API key lives. Numbers a machine can read (mirroring, length) are
printed; the semantic call is made locally.

Then it re-measures the campaigns under the frozen bilingual prompt: every
anchor so far was taken with the old system prompt, and gates compare only at
identical prompt.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

# Paths differ per runner (RunPod pod vs Colab VM); one job, two launchers.
ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out"))

sys.path.insert(0, str(ROOT / "python"))
MODEL = "lfm2/LiquidAI/LFM2.5-Audio-1.5B"
TASK = "python/lfm2_audio/inspect_bridge/task.py@voice_eval"
N_QUESTIONS = 20


def run(args: list[str], log_name: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / log_name).open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(args)}\n")
        handle.flush()
        return subprocess.run(args, cwd=str(ROOT), stdout=handle, stderr=handle, check=False).returncode


def questions(path: Path, limit: int) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[:limit]:
        case = json.loads(line)
        out.append((case["id"], next(t["text"] for t in case["turns"] if t["role"] == "user")))
    return out


def decoding_modes() -> None:
    from lfm2_audio.core.prompt import BILINGUAL_SYSTEM
    from lfm2_audio.ds.generation_config import GenerationConfig
    from lfm2_audio.scorer.text.lang_match import detect_language
    from lfm2_audio.serving.model import LFM2Audio

    cases = questions(ROOT / "benchmark/fr_s2s/questions.jsonl", N_QUESTIONS)
    results: dict[str, dict[str, object]] = {}

    for mode, generation in (
        ("text_only", GenerationConfig(text_only=True)),
        ("interleaved", GenerationConfig()),
    ):
        model = LFM2Audio.from_pretrained(
            "LiquidAI/LFM2.5-Audio-1.5B",
            backend="liquid",
            system=BILINGUAL_SYSTEM,
            generation=generation,
        )
        replies, french, lengths = [], 0, []
        with model:
            for case_id, prompt in cases:
                model.reset()
                reply = model.reply(text=prompt)
                text = reply.text or ""
                replies.append({"id": case_id, "question": prompt, "text": text})
                french += int(detect_language(text) == "fr")
                lengths.append(len(text))
        results[mode] = {
            "n": len(replies),
            "french_replies": french,
            "median_chars": statistics.median(lengths) if lengths else None,
            "replies": replies,
        }
        (OUT / "decoding_modes.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{mode}: {french}/{len(replies)} en français, médiane {results[mode]['median_chars']} car.", flush=True)

    print("===RESULT decoding_modes.json===", flush=True)
    print((OUT / "decoding_modes.json").read_text(encoding="utf-8"), flush=True)
    print("===END===", flush=True)


def rebaseline(only: set[str] | None = None) -> None:
    """Re-measure the anchors under the frozen prompt; the old ones used another.

    ``only`` restricts the run to named campaigns. A pod was reclaimed with one
    campaign left to go, and re-running the two that had already reported would
    have been paid twice for nothing.
    """
    inspect_bin = shutil.which("inspect") or "inspect"
    campaigns = [
        ("lang_mirror", "benchmark/lang_mirror/questions.jsonl", "lang_match,utmos", "en"),
        ("fr_s2s", "benchmark/fr_s2s/questions.jsonl", "wer,utmos,dnsmos,lang_match", "fr"),
        ("baseline_en", "benchmark/baseline_en/questions.jsonl", "wer,utmos,dnsmos,lang_match", "en"),
    ]
    if only:
        campaigns = [c for c in campaigns if c[0] in only]
    for name, questions_path, scorers, lang in campaigns:
        log_dir = OUT / "rebaseline" / name
        if log_dir.exists():
            continue
        run(
            [
                inspect_bin,
                "eval",
                TASK,
                "--model",
                MODEL,
                "-M",
                "system=bilingual",
                "-T",
                f"questions={questions_path}",
                "-T",
                f"scorers={scorers}",
                "-T",
                f"asr_language={lang}",
                "--max-samples",
                "1",
                "--log-dir",
                str(log_dir),
            ],
            f"rebaseline_{name}.log",
        )
        summarise_rebaseline()


def summarise_rebaseline() -> None:
    from inspect_ai.log import read_eval_log

    summary: dict[str, dict[str, float]] = {}
    for log_dir in sorted((OUT / "rebaseline").glob("*")):
        evals = sorted(log_dir.glob("**/*.eval"), key=lambda p: p.stat().st_mtime)
        if not evals:
            continue
        log = read_eval_log(str(evals[-1]))
        scores: dict[str, float] = {}
        if log.results:
            for score in log.results.scores:
                for metric_name, metric in score.metrics.items():
                    scores[f"{score.name}/{metric_name}"] = round(metric.value, 4)
        scores["samples"] = float(len(log.samples or []))
        summary[log_dir.name] = scores
    (OUT / "rebaseline.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("===RESULT rebaseline.json===", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("===END===", flush=True)


def main() -> None:
    """``LFM2_ARGS`` names the steps to run; empty means all of them."""
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = set(sys.argv[1:])
    if not wanted or "modes" in wanted:
        decoding_modes()
        (OUT / "MODES_DONE").write_text("ok\n", encoding="utf-8")
    campaigns = {name for name in wanted if name != "modes"}
    if not wanted or campaigns:
        rebaseline(campaigns or None)
    (OUT / "ALL_DONE").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
