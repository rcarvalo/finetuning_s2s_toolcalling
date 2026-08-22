#!/usr/bin/env bash
# Voxtral TTS batch — runs as the start command of a RunPod pod whose image is
# vllm/vllm-openai:v0.22.0 (the CUDA stack its wheels were built for; the whole
# reason this runs on RunPod and not Colab — see docs/pre_training_review.md §5).
set -euo pipefail

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

python /repo/infra/pod_synth_fresh.py

# Keep the pod alive briefly so logs are readable; the operator deletes it.
sleep 600
