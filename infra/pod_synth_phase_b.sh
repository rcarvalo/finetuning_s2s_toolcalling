#!/usr/bin/env bash
# Phase B corpus synthesis — voice the conversational dialogues with Voxtral.
#
# User turns: the three TRAIN voices, augmented (same distribution as the
# existing corpus). Assistant answer turns: the fixed persona voice the Phase B
# recipe prescribes (configs/training/phase_b_s2s.yaml). Tool-call turns and
# tool results stay text-only by design.
#
# Ships the result to the Hub as one artifact: the voiced JSONL plus a tarball
# of the WAVs, under Rcarvalo/tc-en-voice-agent-v1 (phase_b/).
set -euo pipefail

cd /repo
# The package itself (light core deps), plus what the synthesize CLI imports.
# torchaudio is pinned to the image's torch (same minor) with --no-deps: a
# floating install would drag a NEW torch in and break the vLLM server.
python3 -m pip install -q -e . 2>&1 | tail -1
TORCH_V=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
python3 -m pip install -q --no-deps "torchaudio==${TORCH_V}" 2>&1 | tail -1 || \
    python3 -m pip install -q --no-deps torchaudio 2>&1 | tail -1
python3 -m pip install -q soundfile kokoro httpx 2>&1 | tail -1

# Voxtral server — the fresh-set job starts it after the dispatcher, so this
# path must start its own.
nohup vllm serve mistralai/Voxtral-4B-TTS-2603 --omni \
    --gpu-memory-utilization 0.85 > /voxtral.log 2>&1 &
for _ in $(seq 1 180); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
    sleep 5
done
curl -sf http://localhost:8000/health >/dev/null || { echo "SERVER_DEAD"; tail -40 /voxtral.log; exit 1; }
echo "server healthy"

# The curated source is regenerated, not versioned (2.7 MB): the curator is
# deterministic and both inputs are committed (the s2s corpus + every held-out
# test utterance). Exit code 1 on leakage stops the pod before it spends GPU.
python3 -m lfm2_audio.cli.data.curate \
    --source data/tc_en_s2s.jsonl \
    --held-out data/test_fresh_src.jsonl --held-out data/heldout_tests.jsonl \
    --out data/phase_b_train_src.jsonl --allow-leakage

python3 -m lfm2_audio.cli.data.synthesize --engine voxtral --split train \
    --voices casual_male,casual_female,cheerful_female \
    --assistant-voice neutral_female \
    --dialogues data/phase_b_train_src.jsonl \
    --audio-root data/audio_phase_b \
    --out data/phase_b_train.jsonl \
    --concurrency 8

tar -czf /tmp/phase_b_audio.tar.gz -C data audio_phase_b
python3 - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
for local, remote in (
    ("data/phase_b_train.jsonl", "phase_b/train.jsonl"),
    ("/tmp/phase_b_audio.tar.gz", "phase_b/audio.tar.gz"),
):
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id="Rcarvalo/tc-en-voice-agent-v1", repo_type="dataset")
    print(f"pushed {remote}")
PY
echo PHASE_B_DONE
