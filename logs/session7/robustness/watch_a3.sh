#!/bin/bash
# Session 7: wait for the A3 suite (run_suite PID $1) to end or fail; print the reason.
EV=logs/session7/suite/ablation_a3_can/events.jsonl
while true; do
  if grep -q '"event": "\(end\|suite_error\|reap_error\|low_disk\)"' "$EV"; then echo "EVENT: $(grep '"event": "\(end\|suite_error\|reap_error\|low_disk\)"' "$EV" | tail -1)"; break; fi
  if grep -q '"returncode": [1-9]' "$EV"; then echo "NONZERO: $(grep '"returncode": [1-9]' "$EV" | tail -1)"; break; fi
  if grep -q BELOW logs/session7/suite/disk_guard.log 2>/dev/null; then echo "DISK: $(grep BELOW logs/session7/suite/disk_guard.log | tail -1)"; break; fi
  if ! kill -0 "$1" 2>/dev/null; then echo "GONE: run_suite $1 exited"; tail -2 "$EV"; break; fi
  sleep 60
done
date -u
