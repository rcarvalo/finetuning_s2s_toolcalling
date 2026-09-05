#!/usr/bin/env bash
# Bootstrap of a RunPod batch job. The pod's start command fetches this file
# and runs it; everything else lives in the repo, so a run is described by a
# branch and a job name rather than by a blob pasted into a template.
#
# Contract with the caller (pod env vars):
#   LFM2_BRANCH   branch to check out            (default: main)
#   LFM2_JOB      script under infra/jobs/       (required)
#   LFM2_ARGS     extra args forwarded to it     (optional)
#
# Two things make this remotely drivable without SSH or credentials:
#   * every result is printed between ===RESULT=== markers, so it survives in
#     the pod logs even if the pod dies right after;
#   * artifacts are served over the pod's HTTP proxy port, so they can be
#     fetched with plain curl instead of being pushed anywhere.
set -uo pipefail

BRANCH="${LFM2_BRANCH:-main}"
JOB="${LFM2_JOB:?LFM2_JOB manquant}"
REPO_URL="https://github.com/rcarvalo/finetuning_s2s_toolcalling.git"
ROOT=/workspace/repo
OUT=/workspace/out

mkdir -p "$OUT"
echo "=== bootstrap: branche $BRANCH, job $JOB"

# The RunPod image exports HF_HUB_ENABLE_HF_TRANSFER=1 without shipping the
# package, so every Hub download dies with "hf_transfer is enabled but not
# available" — including the model weights a job exists to run. Install it
# (it is also genuinely faster on multi-GB checkpoints) and fall back to
# disabling the flag if that fails.
pip install -q hf_transfer 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0

# Artifact retrieval starts BEFORE the job: if the job crashes, its logs and
# whatever it wrote are still downloadable.
python -m http.server 8000 --directory "$OUT" >/dev/null 2>&1 &

if [ ! -f "$ROOT/pyproject.toml" ]; then
  rm -rf "$ROOT"
  git clone --branch "$BRANCH" "$REPO_URL" "$ROOT" || { echo "=== ÉCHEC: clone"; sleep infinity; }
fi
cd "$ROOT"
git fetch origin "$BRANCH" && git reset --hard "origin/$BRANCH"
echo "=== commit: $(git log --oneline -1)"

# pytest is a dev dependency, absent from the runtime extras — and the
# assertion below is not optional, so it is installed here rather than skipped.
pip install -q -e ".[serving-liquid,eval,inspect]" pytest 2>&1 | tail -5

# Behavioural assertion, not a version check: a campaign once replayed stale
# code and published wrong numbers.
if ! python -m pytest tests/test_inspect_bridge.py -k "target_text or audio_only" -q 2>&1 | tail -3; then
  echo "=== ÉCHEC: tests de non-régression du pont — code périmé, on ne mesure rien"
  sleep infinity
fi

JOB_PATH="infra/jobs/${JOB}.py"
[ -f "$JOB_PATH" ] || { echo "=== ÉCHEC: $JOB_PATH introuvable"; sleep infinity; }

# Two guards for a pod nobody is watching, both opt-in through the env:
#   LFM2_MAX_HOURS   hard wall-clock cap on the job (a resumable job loses
#                    nothing; a hung one stops costing money)
#   LFM2_AUTO_DELETE "1" deletes THIS pod once the job is over, through the
#                    REST API — needs RUNPOD_API_KEY in the pod env. Pods left
#                    running overnight cost ~5 $ for nothing on 28/08.
echo "=== démarrage du job $JOB (plafond ${LFM2_MAX_HOURS:-aucun} h, auto-suppression ${LFM2_AUTO_DELETE:-0})"
if [ -n "${LFM2_MAX_HOURS:-}" ]; then
  timeout --signal=TERM --kill-after=120 "${LFM2_MAX_HOURS}h" python -u "$JOB_PATH" ${LFM2_ARGS:-} 2>&1 | tee "$OUT/${JOB}.log"
else
  python -u "$JOB_PATH" ${LFM2_ARGS:-} 2>&1 | tee "$OUT/${JOB}.log"
fi
STATUS=${PIPESTATUS[0]}
echo "=== job terminé, code $STATUS"

delete_this_pod() {
  curl -sS -X DELETE "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}" \
       -H "Authorization: Bearer ${RUNPOD_API_KEY}" && echo "=== pod supprimé"
}
if [ "${LFM2_AUTO_DELETE:-0}" = "1" ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  if [ "$STATUS" -ne 0 ]; then
    echo "=== échec : artefacts servis 30 min sur le port 8000, puis suppression"
    sleep 1800
  fi
  echo "=== auto-suppression du pod ${RUNPOD_POD_ID}"
  delete_this_pod
  sleep 300
fi

# Stay up so the artifacts remain fetchable; the operator deletes the pod,
# which is also what stops the billing.
echo "=== artefacts servis sur le port 8000 — supprimer le pod quand ils sont récupérés"
sleep infinity
