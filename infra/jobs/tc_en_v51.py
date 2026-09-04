#!/usr/bin/env python3
"""EN tool-calling v5.1 on Colab: voice the refusals, pack, train — one stage per cell.

v5 failed its gate on self-contradictory refusals; v5.1 changes ONE thing, the
strict guard in `data_prep/contextual_miss.py` (docs/v5_report.md). The corpus
transform already ran on the laptop and its outputs sit on the Hub
(`phase_b/train_v5_1.jsonl`, `phase_b/miss_to_tts_v5_1.jsonl`); the stages
here are the ones that need a GPU or liquid-audio.

    LFM2_JOB=tc_en_v51 LFM2_ARGS="--stage voice"   # Qwen3-TTS, Aiden, 225 clips
    LFM2_JOB=tc_en_v51 LFM2_ARGS="--stage pack"    # rebuild + pack from the Hub
    LFM2_JOB=tc_en_v51 LFM2_ARGS="--stage push"    # packed tensors -> Rcarvalo/tc-en-v5_1-packed
    LFM2_JOB=tc_en_v51 LFM2_ARGS="--stage train"   # optional: LoRA, adapter pushed every 200 steps

Every stage is resumable from what the Hub already holds, because a Colab VM
does not survive: the voice stage restores its partial tarball, the pack stage
re-fetches everything, and a preempted training warm-starts from the last
pushed adapter with `--resume-after N` (the v4 pattern: init_adapter, the
remaining steps, a short warmup).

Run the voice stage, then restart the runtime before pack/train: qwen-tts pulls
its own transformers, and sharing one Python with the trainer is a risk this
job does not need to take.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("LFM2_ROOT", "/content/finetuning_s2s_toolcalling"))
OUT = Path(os.environ.get("LFM2_OUT", "/content/out"))
TAG = os.environ.get("TC_EN_TAG", "v5_1")
CONFIG = ROOT / "configs/training/tc_en_voice_agent_v5_1.yaml"
TOTAL_STEPS = 1970
"""What v5 ran under the same recipe (1.5 epochs on 5258 examples)."""
HUB_ADAPTER = "Rcarvalo/lfm25-tc-en-v5_1-adapter"


def run(cmd: list[str], **env: str) -> None:
    print("===", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env={**os.environ, "TC_EN_TAG": TAG, **env}, check=True)


def stage_voice() -> None:
    # qwen-tts drags torchao 0.10; peft refuses anything below 0.16, and 0.16+
    # segfaults on this Python. Absent, peft skips it — so it goes.
    run([sys.executable, "-m", "pip", "install", "-q", "qwen-tts"])
    run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    run([sys.executable, "infra/pod_synth_misses_v5.py"], ASSISTANT_VOICE=os.environ.get("ASSISTANT_VOICE", "Aiden"))
    print(f"===RESULT=== stage=voice tag={TAG} status=done", flush=True)


def stage_pack() -> None:
    run([sys.executable, "infra/prepare_v5.py", "--stage", "pack"])
    for split in ("train", "val"):
        dataset = ROOT / f"datasets/tc_en_{TAG}_{split}"
        if not dataset.exists():
            raise SystemExit(f"packed dataset missing: {dataset}")
    print(f"===RESULT=== stage=pack tag={TAG} status=done", flush=True)


PACKED_REPO = os.environ.get("TC_EN_PACKED_REPO", f"Rcarvalo/tc-en-{TAG}-packed")
"""Where the packed, training-ready tensors go. A dataset on the Hub is the
deliverable; training is what one does with it later, on any machine."""


def stage_push() -> None:
    from datasets import load_from_disk

    for split in ("train", "val"):
        dataset = ROOT / f"datasets/tc_en_{TAG}_{split}"
        if not dataset.exists():
            raise SystemExit(f"nothing to push: {dataset} missing — run --stage pack first")
        load_from_disk(str(dataset)).push_to_hub(PACKED_REPO, split=split, private=True)
        print(f"pushed {dataset} -> {PACKED_REPO}:{split}", flush=True)
    print(f"===RESULT=== stage=push tag={TAG} repo={PACKED_REPO} status=done", flush=True)


def derive_resume_config(base: dict, steps_done: int) -> dict:
    """The v4 resume recipe: warm start from the pushed adapter, finish the remaining steps.

    A short warmup on purpose — the model is already warm, and replaying 150
    warmup steps would push the LR back up on a run that had settled.
    """
    config = dict(base)
    config["lora"] = {**config.get("lora", {}), "init_adapter": config["hub_repo"]}
    config.pop("num_epochs", None)
    config["max_steps"] = max(TOTAL_STEPS - steps_done, 1)
    config["warmup_steps"] = 30
    config["wandb_run_name"] = f"{config.get('wandb_run_name', 'tc_en_v5_1')}_resume_{steps_done}"
    return config


def stage_train(resume_after: int | None) -> None:
    config_path = CONFIG
    if resume_after is not None:
        derived = derive_resume_config(yaml.safe_load(CONFIG.read_text(encoding="utf-8")), resume_after)
        config_path = OUT / f"tc_en_{TAG}_resume_{resume_after}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(derived, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"=== resume config: {config_path} (max_steps={derived['max_steps']})", flush=True)
    run([sys.executable, "-m", "lfm2_audio.cli.train.sft", "--config", str(config_path)])
    print(f"===RESULT=== stage=train tag={TAG} adapter={HUB_ADAPTER} status=done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["voice", "pack", "push", "train", "all"], required=True)
    parser.add_argument("--resume-after", type=int, default=None, help="steps already trained (Hub adapter)")
    args = parser.parse_args()

    if args.stage in ("voice", "all"):
        stage_voice()
    if args.stage in ("pack", "all"):
        stage_pack()
    if args.stage in ("push", "all"):
        stage_push()
    # Training is NOT part of `all`: the dataset on the Hub is the deliverable,
    # and a training run is a separate, deliberate decision.
    if args.stage == "train":
        stage_train(args.resume_after)


if __name__ == "__main__":
    main()
