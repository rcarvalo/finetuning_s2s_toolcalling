"""Two design questions, decided by measurement before any A100 hour.

Q1. Can ONE interleaving ratio serve both languages?
    The schedule alternates 6 text tokens then 12 audio frames, and the model's
    French speech carries ~1.7x too much audio for its text (baseline 0B). The
    ratio is read from the config at every generate call, so it is a serving
    setting, not a trained weight. Sweeping it over BOTH languages says whether
    a single compromise exists or whether the ratio must follow the language.

Q2. Can ONE system prompt mirror both languages?
    A French prompt lifted French output from 56% to 82% on the ASR task, but
    that prompt was both written in French AND explicitly asked for French —
    two causes changed at once. The arms below separate the prompt's language
    from the presence of an explicit mirroring rule, and check the control that
    matters: a French prompt must not break English.

Results are written after every arm, not at the end: the previous session was
pruned mid-run and everything in flight was lost.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/repo/python")

ROOT = Path("/workspace/repo")
OUT = Path("/workspace/out")
BRANCH = "rd/pr_rca_eval_baseline"
MODEL = "lfm2/LiquidAI/LFM2.5-Audio-1.5B"
TASK = "python/lfm2_audio/inspect_bridge/task.py@voice_eval"

RATIOS: list[tuple[str, tuple[int, int] | None]] = [
    ("r6x12", (6, 12)),  # shipped, English-calibrated
    ("r6x9", (6, 9)),
    ("r6x7", (6, 7)),
]

# P1 (fr 0.79 / en 0.95 / switch 0.90) and P2 (0.90 / 1.00 / 0.95) are already
# measured — Colab pruned the session that held them, and re-running arms whose
# numbers we already have would cost 30 minutes for nothing.
# Measured: P1 fr 0.79 / en 0.95 / switch 0.90 · P2 0.90 / 1.00 / 0.95
# · P3 0.94 / 1.00 / 0.90. Nothing left to run here — the sweep below is what
# this job still owes.
PROMPT_ARMS: dict[str, str] = {}


def run(args: list[str], log_name: str, cwd: Path = ROOT) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / log_name).open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(args)}\n")
        handle.flush()
        return subprocess.run(args, cwd=str(cwd), stdout=handle, stderr=handle, check=False).returncode


# --------------------------------------------------------------------------- #
# Q2 — system prompts
# --------------------------------------------------------------------------- #


def prompt_grid() -> None:
    inspect_bin = shutil.which("inspect") or "inspect"
    for name, system in PROMPT_ARMS.items():
        log_dir = OUT / "prompts" / name
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
                f"system={system}",
                "-T",
                "questions=benchmark/lang_mirror/questions.jsonl",
                "-T",
                "scorers=lang_match",
                "--max-samples",
                "1",
                "--log-dir",
                str(log_dir),
            ],
            f"prompt_{name}.log",
        )
        summarise_prompts()


def summarise_prompts() -> None:
    from inspect_ai.log import read_eval_log

    summary = {}
    for log_dir in sorted((OUT / "prompts").glob("*")):
        evals = sorted(log_dir.glob("**/*.eval"), key=lambda p: p.stat().st_mtime)
        if not evals:
            continue
        log = read_eval_log(str(evals[-1]))
        groups: dict[str, list[float]] = {"fr": [], "en": [], "switch": []}
        for sample in log.samples or []:
            score = (sample.scores or {}).get("lang_match")
            if score is None or not isinstance(score.value, (int, float)) or score.value < 0:
                continue
            sid = str(sample.id)
            subset = "switch" if "switch" in sid else ("fr" if "_fr_" in sid else "en")
            groups[subset].append(float(score.value))
        summary[log_dir.name] = {
            subset: {"n": len(v), "mirror_rate": round(statistics.mean(v), 3) if v else None}
            for subset, v in groups.items()
        }
    (OUT / "prompt_grid.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("prompts:", json.dumps(summary, ensure_ascii=False), flush=True)


# --------------------------------------------------------------------------- #
# Q1 — interleaving ratio, both languages
# --------------------------------------------------------------------------- #


def snapshot_dir() -> Path:
    """Where the base checkpoint actually lives, asked rather than guessed.

    Globbing ``/root/.cache/huggingface/hub`` worked on Colab and found nothing
    on RunPod, whose cache sits elsewhere — and the failure came at the END of
    a paid run. ``snapshot_download`` returns the real path and is a no-op when
    the files are already there.
    """
    from huggingface_hub import snapshot_download

    return Path(snapshot_download("LiquidAI/LFM2.5-Audio-1.5B"))


def variant_dir(name: str, ratio: tuple[int, int] | None, source: Path) -> Path:
    target = OUT / f"ckpt_{name}"
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == "config.json":
            continue
        link = target / item.name
        if not (link.exists() or link.is_symlink()):
            link.symlink_to(item.resolve())
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if ratio is not None:
        config["interleaved_n_text"], config["interleaved_n_audio"] = ratio
    (target / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    return target


def questions(path: Path, limit: int) -> list[str]:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines()[:limit]:
        case = json.loads(line)
        texts.append(next(t["text"] for t in case["turns"] if t["role"] == "user"))
    return texts


def ratio_sweep() -> None:
    import soundfile as sf

    from lfm2_audio.scorer.text.lang_match import detect_language
    from lfm2_audio.serving.model import LFM2Audio

    source = snapshot_dir()
    sets = {
        "fr": questions(ROOT / "benchmark/fr_s2s/questions.jsonl", 20),
        "en": questions(ROOT / "benchmark/baseline_en/questions.jsonl", 20),
    }
    results_path = OUT / "ratio_sweep.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}

    for name, ratio in RATIOS:
        if name in results:
            continue
        model = LFM2Audio.from_pretrained(variant_dir(name, ratio, source), backend="liquid")
        entry = {}
        with model:
            for lang, prompts in sets.items():
                rates, mirrored = [], 0
                audio_dir = OUT / "audio" / name / lang
                audio_dir.mkdir(parents=True, exist_ok=True)
                for index, prompt in enumerate(prompts):
                    model.reset()
                    reply = model.reply(text=prompt)
                    text, audio = reply.text or "", reply.audio
                    if audio is None or len(text) < 20:
                        continue
                    rates.append((len(audio.samples) / audio.sample_rate) / len(text))
                    mirrored += int(detect_language(text) == lang)
                    sf.write(str(audio_dir / f"q{index:02d}.wav"), audio.samples, audio.sample_rate, subtype="PCM_16")
                    (audio_dir / f"q{index:02d}.txt").write_text(text, encoding="utf-8")
                entry[lang] = {
                    "n": len(rates),
                    "sec_per_char_median": round(statistics.median(rates), 4) if rates else None,
                    "mirrored": mirrored,
                }
        results[name] = entry
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(name, entry, flush=True)


def _emit_results() -> None:
    """Print every result between markers.

    A pod can vanish before its files are fetched — Colab pruned two sessions
    in an hour and took the numbers with them. Printed results survive in the
    pod logs, which outlive the pod.
    """
    for name in ("prompt_grid.json", "ratio_sweep.json"):
        path = OUT / name
        if path.exists():
            print(f"===RESULT {name}===", flush=True)
            print(path.read_text(encoding="utf-8"), flush=True)
            print("===END===", flush=True)


def main() -> None:
    """Checkout and install are the entrypoint's job; this only measures."""
    OUT.mkdir(parents=True, exist_ok=True)
    prompt_grid()
    (OUT / "PROMPTS_DONE").write_text("ok\n", encoding="utf-8")
    ratio_sweep()
    (OUT / "ALL_DONE").write_text("ok\n", encoding="utf-8")
    _emit_results()


if __name__ == "__main__":
    main()
