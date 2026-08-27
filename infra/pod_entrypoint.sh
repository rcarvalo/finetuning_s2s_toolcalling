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

pip install -q -e ".[serving-liquid,eval,inspect]" 2>&1 | tail -5

# Behavioural assertion, not a version check: a campaign once replayed stale
# code and published wrong numbers.
if ! python -m pytest tests/test_inspect_bridge.py -k "target_text or audio_only" -q 2>&1 | tail -3; then
  echo "=== ÉCHEC: tests de non-régression du pont — code périmé, on ne mesure rien"
  sleep infinity
fi

JOB_PATH="infra/jobs/${JOB}.py"
[ -f "$JOB_PATH" ] || { echo "=== ÉCHEC: $JOB_PATH introuvable"; sleep infinity; }

echo "=== démarrage du job $JOB"
python -u "$JOB_PATH" ${LFM2_ARGS:-} 2>&1 | tee "$OUT/${JOB}.log"
STATUS=${PIPESTATUS[0]}
echo "=== job terminé, code $STATUS"

# Stay up so the artifacts remain fetchable; the operator deletes the pod,
# which is also what stops the billing.
echo "=== artefacts servis sur le port 8000 — supprimer le pod quand ils sont récupérés"
sleep infinity
