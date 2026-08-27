#!/usr/bin/env bash
# Colab twin of pod_entrypoint.sh: same contract, different machine.
#
#   LFM2_BRANCH   branch to check out              (default: main)
#   LFM2_JOB      script under infra/jobs/         (required)
#   LFM2_ARGS     extra args forwarded to it       (optional)
#
# Run this IN THE FOREGROUND of a `colab exec`, not detached.
#
# Measured on 27/08: a five-minute cell ran to completion although the client
# gave up after about one minute — `colab exec` timing out does not stop the
# kernel. Detaching was therefore never needed, and it was actively harmful:
# Colab's activity signal is kernel occupancy, so a detached job leaves the
# kernel idle and the session looks abandoned. Three sessions were reclaimed
# that day, one of them seconds after finishing its work.
#
# The working shape is one chunk of work per exec call: the kernel is busy
# while it runs (session stays alive), the call returns (or the client times
# out, harmlessly), artifacts are downloaded, and the next call resumes.
# Jobs must therefore skip what they have already produced.
#
# Note that probes queue behind a busy kernel — also measured — so do not
# expect to observe a chunk while it runs. Observe between chunks.
#
# Results are still printed between ===RESULT=== markers: they survive in the
# client output even when the VM does not. Files do not, and `colab download`
# needs the VM alive — the one thing a RunPod pod, with its HTTP port, does
# better.
set -uo pipefail

BRANCH="${LFM2_BRANCH:-main}"
JOB="${LFM2_JOB:?LFM2_JOB manquant}"
export LFM2_ROOT="${LFM2_ROOT:-/content/finetuning_s2s_toolcalling}"
export LFM2_OUT="${LFM2_OUT:-/content/out}"
REPO_URL="https://github.com/rcarvalo/finetuning_s2s_toolcalling.git"

mkdir -p "$LFM2_OUT"
echo "=== bootstrap Colab : branche $BRANCH, job $JOB"

if [ ! -f "$LFM2_ROOT/pyproject.toml" ]; then
  rm -rf "$LFM2_ROOT"
  git clone --branch "$BRANCH" "$REPO_URL" "$LFM2_ROOT" || { echo "=== ÉCHEC: clone"; exit 1; }
fi
cd "$LFM2_ROOT" || exit 1
git fetch origin "$BRANCH" && git reset --hard "origin/$BRANCH"
echo "=== commit: $(git log --oneline -1)"

# pytest is a dev dependency, absent from the runtime extras, and the assertion
# below is not optional.
python -m pip install -q -e ".[serving-liquid,eval,inspect]" pytest 2>&1 | tail -3

# Behavioural assertion, not a version check: a campaign once replayed stale
# code and published wrong numbers.
if ! python -m pytest tests/test_inspect_bridge.py -k "target_text or audio_only" -q 2>&1 | tail -2; then
  echo "=== ÉCHEC: tests de non-régression du pont — code périmé, on ne mesure rien"
  exit 1
fi

JOB_PATH="infra/jobs/${JOB}.py"
[ -f "$JOB_PATH" ] || { echo "=== ÉCHEC: $JOB_PATH introuvable"; exit 1; }

echo "=== démarrage du job $JOB"
python -u "$JOB_PATH" ${LFM2_ARGS:-} 2>&1
echo "=== job terminé, code $?"
