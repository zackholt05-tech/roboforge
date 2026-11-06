#!/usr/bin/env bash
# Run a RoboForge command on the canonical evaluation platform (DECISIONS D-017):
# the Linux aarch64 image roboforge-sim-cpu, pinned by image ID in configs/eval/platform.yaml.
#
#   infra/containers/run_sim_cpu.sh python scripts/check_determinism.py
#
# The host working tree's code, configs, committed data artifacts and docs are bind-mounted
# READ-ONLY; only reports/session<N>, logs/session<N> and runs are writable, where N is
# ROBOFORGE_EVIDENCE_SESSION (default 2), e.g.
#   ROBOFORGE_EVIDENCE_SESSION=3 make in-sim CMD="python scripts/smoke_sim.py --task lift ..."
# Earlier sessions' evidence stays read-only or unmounted. N is also passed into the container
# as ROBOFORGE_EVIDENCE_SESSION, from which the evidence scripts derive their default output
# directories (DECISIONS D-039). The script refuses to
# run if the local image ID differs from the pinned one. The host git revision and dirty flag
# are passed in, because the container has no .git directory. Commands run with the image's
# own locked venv (/app/.venv, no `uv sync` at run time, so the platform never changes) and the
# mounted source first on PYTHONPATH.
set -euo pipefail
cd "$(dirname "$0")/../.."
IMAGE="${ROBOFORGE_IMAGE:-roboforge-sim-cpu:session0}"
PINNED=$(sed -n 's/^image_id: *//p' configs/eval/platform.yaml)
ACTUAL=$(docker image inspect -f '{{.Id}}' "$IMAGE")
if [[ "$ACTUAL" != "$PINNED" ]]; then
  echo "image $IMAGE is $ACTUAL, but configs/eval/platform.yaml pins $PINNED" >&2
  exit 2
fi
DIRTY=0
if [[ -n "$(git status --porcelain -- . ':(exclude)reports' ':(exclude)logs' ':(exclude)runs')" ]]; then
  DIRTY=1
fi
S="${ROBOFORGE_EVIDENCE_SESSION:-2}"
if [[ ! "$S" =~ ^[0-9]+$ ]]; then
  echo "ROBOFORGE_EVIDENCE_SESSION must be a session number, got '$S'" >&2
  exit 2
fi
mkdir -p "reports/session$S" runs "logs/session$S"
ro() { printf -- "-v %s:/app/%s:ro " "$PWD/$1" "$1"; }
# Eval-state generation (scripts/generate_eval_states.py without --check) needs a writable
# data/eval_states; every other use mounts it read-only.
EVAL_MODE=ro
if [[ "${ROBOFORGE_WRITE_EVAL_STATES:-0}" == "1" ]]; then EVAL_MODE=rw; fi
# shellcheck disable=SC2046
exec docker run --rm \
  $(ro src) $(ro scripts) $(ro configs) $(ro tests) $(ro docs) \
  $(ro data/manifests) $(ro data/splits) $(ro data/normalization) \
  -v "$PWD/data/eval_states:/app/data/eval_states:$EVAL_MODE" \
  $(ro data/raw) $(ro data/processed) \
  -v "$PWD/reports/session$S:/app/reports/session$S" -v "$PWD/runs:/app/runs" \
  -v "$PWD/logs/session$S:/app/logs/session$S" \
  -v "$PWD/reports/feasibility:/app/reports/feasibility:ro" \
  -e ROBOFORGE_CONTAINER_IMAGE="$IMAGE" -e ROBOFORGE_CONTAINER_IMAGE_ID="$ACTUAL" \
  -e ROBOFORGE_GIT_SHA="$(git rev-parse HEAD)" -e ROBOFORGE_GIT_DIRTY="$DIRTY" \
  -e ROBOFORGE_EVIDENCE_SESSION="$S" \
  -e PYTHONPATH=/app/src -e PATH=/app/.venv/bin:/usr/local/bin:/usr/bin:/bin \
  "$IMAGE" "$@"
