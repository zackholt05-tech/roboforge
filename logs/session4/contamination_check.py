"""Session 4 contamination check over every Session 4 rollout record (run after the evals).

Counts record partitions and state IDs under runs/s4*__val_*, lists the src/scripts files
that name the locked suite (expected: the same 3 as Session 3), and confirms that
docs/FINAL_TEST_FREEZE.md does not exist. Reads records only; loads no evaluation state.
"""
import collections
import json
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parents[2]
parts, ids = collections.Counter(), set()
for rec in sorted(root.glob("runs/s4*__val_*/episodes/*/record.json")):
    s = json.loads(rec.read_text())["episode"]["initial_state"]
    parts[s["partition"]] += 1
    ids.add(s["state_id"])
print("records:", sum(parts.values()), "partitions:", dict(parts))
print("distinct state_ids:", len(ids), "all lift_validation_*:",
      all(i.startswith("lift_validation_") for i in ids))
locked = "final" + "_test"
hits = subprocess.run(["git", "grep", "-l", locked, "--", "src", "scripts"], cwd=root,
                      capture_output=True, text=True).stdout.split()
print("src/scripts files naming the locked suite:", hits)
print("FINAL_TEST_FREEZE.md exists:", (root / "docs" / "FINAL_TEST_FREEZE.md").exists())
