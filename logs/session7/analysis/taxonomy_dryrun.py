"""Taxonomy dry run (RESEARCH_PROTOCOL §14.5) on every Can validation summary in
reports/session7/validation with the committed taxonomy code: one label per episode, taxonomy
successes equal to each summary's successes. Read-only; writes one JSON under logs/session7/analysis.
A first pass on 440 summaries (2026-11-08 ~14:30 UTC, suite still running) gave the same
taxonomy hash and no problems; this file is the pass on all 480."""
import collections, glob, json, sys
from pathlib import Path
from roboforge.eval import taxonomy as tx
from roboforge.paths import REPO_ROOT as MAIN
out = MAIN / "logs/session7/analysis/taxonomy_dryrun_can_validation.json"
rows, totals, problems = [], collections.Counter(), []
for f in sorted(glob.glob(str(MAIN / "reports/session7/validation/*.json"))):
    s = json.loads(Path(f).read_text())
    labels = tx.classify_records(MAIN / s["records_dir"])
    n_ok = len(labels) == s["n_episodes"]
    succ = sum(l["category"] == "success" for l in labels)
    one = all(isinstance(l.get("category"), str) for l in labels)
    if not (n_ok and succ == s["successes"] and one):
        problems.append({"summary": Path(f).name, "n_labels": len(labels), "n_episodes": s["n_episodes"],
                         "taxonomy_successes": succ, "summary_successes": s["successes"]})
    c = tx.counts(labels); totals.update(c)
    rows.append({"summary": Path(f).name, "counts": c})
rep = {"taxonomy_sha256": tx.taxonomy_sha256(), "n_summaries": len(rows),
       "n_episodes": sum(totals.values()), "totals": dict(totals), "problems": problems, "rows": rows}
out.write_text(json.dumps(rep, indent=1) + "\n")
print(json.dumps({k: rep[k] for k in ("taxonomy_sha256", "n_summaries", "n_episodes", "totals")}, indent=1))
print("problems:", len(problems), problems[:3])
