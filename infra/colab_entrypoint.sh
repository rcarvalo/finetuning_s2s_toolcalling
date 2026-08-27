#!/usr/bin/env bash
# Colab twin of pod_entrypoint.sh: same contract, different machine.
#
#   LFM2_BRANCH   branch to check out              (default: main)
#   LFM2_JOB      script under infra/jobs/         (required)
#   LFM2_ARGS     extra args forwarded to it       (optional)
#
# Colab prunes sessions without warning — twice in an hour on 27/08, each time
# taking a running campaign. Two consequences are built in here rather than
# rediscovered:
#
#   * results are printed between ===RESULT=== markers, so the numbers survive
#     in the probe output even when the VM does not;
#   * the job is re-entered, not restarted: jobs write their outputs as they go
#     and skip what is already done, so a pruned session costs one shard.
#
# Artifacts come back with `colab download`; there is no public port here, which
# is the one thing a RunPod pod does better.
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
