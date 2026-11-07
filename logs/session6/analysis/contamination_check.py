"""Session 6 contamination check (reads records and scripts only; loads no evaluation state).

* every Session 6 rollout record under runs/s6_*__val_* is in partition eval_validation, and
  every Session 6 validation summary names the validation suite's content hash;
* every summary in reports/session6/validation is canonical, complete and clean (git dirty
  false) and belongs to a planned s6 run;
* the scripts added in Session 6 load only the validation suite or none;
* the src/scripts files naming the locked suite are the same 3 as in Sessions 3-5;
* docs/FINAL_TEST_FREEZE.md does not exist.

    uv run --frozen python logs/session6/analysis/contamination_check.py
"""
import collections
import json
import pathlib
import re
import subprocess

root = pathlib.Path(__file__).resolve().parents[3]
parts, ids, n_rec = collections.Counter(), set(), 0
for rec in sorted(root.glob("runs/s6_*__val_*/episodes/*/record.json")):
    s = json.loads(rec.read_text())["episode"]["initial_state"]
    parts[s["partition"]] += 1
    ids.add(s["state_id"])
    n_rec += 1
print("records:", n_rec, "partitions:", dict(parts))
print("distinct state_ids:", len(ids), "all lift_validation_*:",
      all(i.startswith("lift_validation_") for i in ids))
summ = sorted((root / "reports/session6/validation").glob("*.json"))
bad = []
suites, dirty = collections.Counter(), collections.Counter()
for f in summ:
    s = json.loads(f.read_text())
    suites[(s["suite"], s["partition"], s["suite_manifest_content_sha256"][:8])] += 1
    dirty[(s["git"]["sha"][:7], s["git"]["dirty"])] += 1
    if not (s["canonical_platform"] and s["eligible_for_selection"] and not s["partial"]
            and s["n_episodes"] == 50 and s["verify_passed"] and s["n_errors"] == 0
            and s["train_run_id"].startswith("s6_")):
        bad.append(f.name)
print("summaries:", len(summ), "suites:", dict(suites), "git:", dict(dirty))
print("summaries failing canonical/complete/verified/no-error/s6:", bad)
for f in ("scripts/run_suite.py", "scripts/plot_validation.py", "scripts/render_episodes.py",
          "src/roboforge/train/suite.py", "src/roboforge/eval/tuning.py"):
    p = root / f
    if p.exists():
        found = sorted(set(re.findall(r'load_suite\([^,]+,\s*"(\w+)"', p.read_text())))
        print(f"{f}: load_suite suites {found}")
locked = "final" + "_test"
hits = subprocess.run(["git", "grep", "-l", locked, "--", "src", "scripts"], cwd=root,
                      capture_output=True, text=True).stdout.split()
print("src/scripts files naming the locked suite:", hits)
print("FINAL_TEST_FREEZE.md exists:", (root / "docs" / "FINAL_TEST_FREEZE.md").exists())
