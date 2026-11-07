#!/usr/bin/env bash
# Session 7 disk guard: every 120 s, log free space on / ; if it falls below MIN_GB, send ONE
# SIGINT to the suite orchestrator (graceful stop: no new launches, training runs write a resume
# checkpoint, running evaluations finish; resume later with --resume-interrupted, D-046/D-054).
cd "$(dirname "$0")/../../.."
PLAN="${1:-configs/experiments/level1_core_can.yaml}"; MIN_GB="${2:-3}"; LOG=logs/session7/suite/disk_guard.log
sent=0
while pgrep -f "scripts/run_suite.py --plan $PLAN" >/dev/null; do
  free_kb=$(df -k / | awk 'NR==2 {print $4}'); free_gb=$(echo "scale=2; $free_kb/1048576" | bc)
  echo "$(date -u +%FT%TZ) free_gb=$free_gb runs_gb=$(du -sk runs 2>/dev/null | awk '{printf "%.2f", $1/1048576}')" >> $LOG
  if [ "$sent" = 0 ] && [ "$(echo "$free_gb < $MIN_GB" | bc)" = 1 ]; then
    pid=$(pgrep -f "scripts/run_suite.py --plan $PLAN" | head -1)
    echo "$(date -u +%FT%TZ) FREE SPACE BELOW $MIN_GB GB: SIGINT to suite pid $pid" >> $LOG
    kill -INT "$pid"; sent=1
  fi
  sleep 120
done
