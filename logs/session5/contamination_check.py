"""Session 5 contamination check (reads records and scripts only; loads no evaluation state).

* every Session 5 rollout record under runs/s5*__val_* is in partition eval_validation;
* the rollout profiler and the render probe load only the validation suite or none;
* the src/scripts files naming the locked suite are the same 3 as in Sessions 3-4;
* docs/FINAL_TEST_FREEZE.md does not exist.
"""
import collections
import json
import pathlib
import re
import subprocess

root = pathlib.Path(__file__).resolve().parents[2]
parts, ids = collections.Counter(), set()
for rec in sorted(root.glob("runs/s5*__val_*/episodes/*/record.json")):
    s = json.loads(rec.read_text())["episode"]["initial_state"]
    parts[s["partition"]] += 1
    ids.add(s["state_id"])
print("records:", sum(parts.values()), "partitions:", dict(parts))
print("distinct state_ids:", len(ids), "all *_validation_*:",
      all("_validation_" in i for i in ids))
for f in ("scripts/profile_rollouts.py", "scripts/profile_parallel.py",
          "scripts/render_probe.py", "src/roboforge/compute/remote.py"):
    suites = sorted(set(re.findall(r'load_suite\([^,]+,\s*"(\w+)"', (root / f).read_text())))
    print(f"{f}: load_suite suites {suites}")
locked = "final" + "_test"
hits = subprocess.run(["git", "grep", "-l", locked, "--", "src", "scripts"], cwd=root,
                      capture_output=True, text=True).stdout.split()
print("src/scripts files naming the locked suite:", hits)
print("FINAL_TEST_FREEZE.md exists:", (root / "docs" / "FINAL_TEST_FREEZE.md").exists())
