#!/usr/bin/env bash
# Voxtral TTS batch — runs as the start command of a RunPod pod whose image is
# vllm/vllm-openai:v0.22.0 (the CUDA stack its wheels were built for; the whole
# reason this runs on RunPod and not Colab — see docs/pre_training_review.md §5).
set -euo pipefail

# The baked /repo is a plain COPY (no .git — dockerignore drops it), so a
# refresh means recloning and swapping. This is what lets job-script changes
# reach pods without rebuilding the image; on any failure the baked revision
# runs — never a dead pod.
BRANCH="${REPO_BRANCH:-rd/pr_rca_eval_baseline}"
if command -v git >/dev/null 2>&1; then
    rm -rf /repo.new
    if git clone --branch "$BRANCH" --depth 1 \
        https://github.com/rcarvalo/finetuning_s2s_toolcalling /repo.new; then
        rm -rf /repo && mv /repo.new /repo
        git -C /repo log --oneline -1
    else
        echo "clone failed — running the baked revision"
    fi
else
    echo "git absent — running the baked revision"
fi

# One image, several jobs: TTS_JOB selects what this pod produces.
if [ "${TTS_JOB:-fresh}" = "phase_b" ]; then
    exec bash /repo/infra/pod_synth_phase_b.sh
fi

echo "=== deps ==="
pip install -q vllm-omni==0.22.0 mistral_common soundfile httpx pyarrow \
    numpy pydantic pydantic-settings pyyaml huggingface-hub

echo "=== voxtral up ==="
nohup vllm serve mistralai/Voxtral-4B-TTS-2603 --omni \
    --gpu-memory-utilization 0.85 > /voxtral.log 2>&1 &
for _ in $(seq 1 180); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
    sleep 5
done
curl -sf http://localhost:8000/health >/dev/null || { echo "SERVER_DEAD"; tail -40 /voxtral.log; exit 1; }
echo "server healthy"

python3 /repo/infra/pod_synth_fresh.py

# Keep the pod alive briefly so logs are readable; the operator deletes it.
sleep 600
