"""Lift taxonomy dry run (RESEARCH_PROTOCOL §14.5): failure categories of the chosen trial's
validation episodes per family, from a suite summary given as argv[1]. Read-only;
output logs/session7/analysis/taxonomy_dry_run_lift.json (run 2026-11-07, before any robustness run)."""
import json, sys, collections
from pathlib import Path
from roboforge.eval import taxonomy as tx
from roboforge.paths import REPO_ROOT as MAIN
summ = json.loads((MAIN / sys.argv[1]).read_text())
fams = {}
for fam in summ["families"]:
    ch = fam["chosen_trial"]
    runs = [r for r in fam["runs"] if r["trial"] == ch]
    tot = collections.Counter(); n = 0
    for r in runs:
        for ev in sorted((MAIN / "runs").glob(f"{r['run_id']}__val_step*")):
            labs = tx.classify_records(ev); n += len(labs)
            tot.update(tx.counts(labs))
    fams[fam["family"]] = (n, dict(tot))
    print(fam["family"], n, {k: v for k, v in tot.items() if v})
json.dump(fams, open(sys.argv[2], "w"), indent=1)
