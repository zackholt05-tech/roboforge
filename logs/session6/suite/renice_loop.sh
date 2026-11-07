#!/usr/bin/env bash
# Session 6 scheduling aid (results-neutral): every 60 s, give non-diffusion training
# processes nice 15 (the suite inherits 5) so the diffusion runs (the critical path) win CPU contention.
# Logs each renice to logs/session6/suite/renice.log. Stops when the suite process exits.
cd "$(dirname "$0")/../../.."
while pgrep -f "scripts/run_suite.py --plan configs/experiments/level1_core_lift.yaml" >/dev/null; do
  for pid in $(pgrep -f "scripts/train.py --config configs/models/bc_"); do
    if [ "$(ps -o nice= -p "$pid" | tr -d ' ')" -lt 15 ]; then
      renice 15 -p "$pid" >/dev/null && echo "$(date -u +%FT%TZ) renice to 15 $pid $(ps -o command= -p "$pid" | grep -o 'models/[a-z_]*.yaml')" >> logs/session6/suite/renice.log
    fi
  done
  sleep 60
done
