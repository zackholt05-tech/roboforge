#!/usr/bin/env bash
# Session 7 scheduling aid (results-neutral, as D-059): every 60 s, give non-diffusion training
# processes nice 15 so the diffusion runs (the critical path) win CPU contention on the shared host.
# Logs each renice to logs/session7/suite/renice.log. Stops when the suite process exits.
cd "$(dirname "$0")/../../.."
PLAN="${1:-configs/experiments/level1_core_can.yaml}"
while pgrep -f "scripts/run_suite.py --plan $PLAN" >/dev/null; do
  for pid in $(pgrep -f "scripts/train.py --config configs/models/bc_"); do
    if [ "$(ps -o nice= -p "$pid" | tr -d ' ')" -lt 15 ]; then
      renice 15 -p "$pid" >/dev/null 2>&1
      echo "$(date -u +%FT%TZ) renice to $(ps -o nice= -p "$pid" | tr -d ' ') $pid $(ps -o command= -p "$pid" | grep -o 'models/[a-z_]*.yaml')" >> logs/session7/suite/renice.log
    fi
  done
  sleep 60
done
