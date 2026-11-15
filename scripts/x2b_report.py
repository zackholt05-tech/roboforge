"""X2-b report from the files only: runs, selections, choices and the X2 test configuration.

Development numbers on X2 validation starts, never a result (D-125 to D-127). Reads the
training manifests under ``runs/x2/``, the evaluation summaries of the X2-a pilot and of X2-b
(``summary_dirs`` of ``configs/extensions/x2b.yaml``) and the choice records; applies the
frozen §7 rule per run; writes ``reports/session16/x2/suite/{suite.json,tables.md}`` and,
once every family is chosen and every run selected, ``x2_test_configuration.json``::

    ROBOFORGE_EVIDENCE_SESSION=16 python scripts/x2b_report.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import git_revision, write_json
from roboforge.x2 import suite as xs
from roboforge.x2.host import now_utc
from roboforge.x2.pilot import FAMILIES


def _du(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def _peak_rss(run: xs.PlannedRun) -> int | None:
    """Peak RSS from the run's ``/usr/bin/time -l`` log (X2-b, or the X2-a pilot's)."""
    for p in (REPO_ROOT / "logs" / "session16" / "x2" / "suite" / f"train_{run.run_id}.log",
              REPO_ROOT / "logs" / "session15" / "x2" / "pilot" / f"train_{run.family}.log"):
        if p.exists() and (run.pilot == ("session15" in str(p))):
            m = re.findall(r"(\d+)\s+maximum resident set size", p.read_text())
            return int(m[-1]) if m else None
    return None


def write_report() -> dict[str, Any]:
    """Write the report; return it."""
    cfg = xs.x2b_config()
    plan = xs.x2b_plan()
    summaries = xs.load_summaries()
    out_dir = REPO_ROOT / cfg["out_dir"]
    choices = {}
    for fam in FAMILIES:
        p = out_dir / "choices" / f"{fam}.json"
        if p.exists():
            choices[fam] = json.loads(p.read_text())
    runs = xs.tuning_runs(plan)
    for fam, ch in choices.items():
        runs += xs.replicate_runs(plan, fam, int(ch["chosen"]["trial"]))
    rows = []
    for r in runs:
        m = xs.read_train_manifest(r)
        sel = xs.selection(r, summaries)
        rows.append({"family": r.family, "trial": r.trial, "seed": r.seed, "run_id": r.run_id,
                     "pilot": r.pilot, "config_sha256": r.config_sha256,
                     "status": m and m["status"], "timing": m and m.get("timing"),
                     "git": m and m.get("git"), "peak_rss_bytes": _peak_rss(r),
                     "disk_bytes": _du(r.run_dir),
                     "n_evaluated": sum(k[0] == r.run_id for k in summaries),
                     "selection": sel and {k: sel[k] for k in ("chosen", "n_tied_on_success")},
                     "validation_curve": sorted((k[1], s["successes"]) for k, s in
                                                summaries.items() if k[0] == r.run_id)})
    new_evals = [s for k, s in summaries.items()
                 if s.get("records_dir", "").startswith("runs/x2/evals")
                 and any(row["run_id"] == k[0] for row in rows)]
    x2b_dir = REPO_ROOT / cfg["summary_dirs"][-1]
    x2b_walls = [json.loads(p.read_text())["wall_s"] for p in sorted(x2b_dir.glob("*.json"))]
    tc = xs.x2_test_configuration(plan, choices, summaries)
    report = {"schema": "roboforge.x2b_report/1", "decisions": cfg["decisions"],
              "label": cfg["label"], "created_utc": now_utc(), "git": git_revision(),
              "plan": cfg["plan"], "pilot_mapping": xs.check_pilot_mapping(plan),
              "runs": rows, "choices": choices,
              "evaluations": {"total_summaries": len(new_evals),
                              "x2b_summaries": len(x2b_walls),
                              "x2b_container_hours": round(sum(x2b_walls) / 3600, 2),
                              "x2b_mean_wall_s": (round(sum(x2b_walls) / len(x2b_walls), 1)
                                                  if x2b_walls else None)},
              "disk_bytes_runs_x2": _du(xs.RUNS),
              "complete": tc is not None}
    write_json(out_dir / "suite.json", report)
    if tc is not None:
        write_json(out_dir / "x2_test_configuration.json", tc)
    (out_dir / "tables.md").write_text(tables(report, tc))
    print(json.dumps({"complete": report["complete"], "chosen": {
        f: c["chosen"]["trial"] for f, c in choices.items()},
        "out": repo_relative(out_dir / "suite.json")}, indent=1))
    return report


def tables(report: dict[str, Any], tc: dict[str, Any] | None) -> str:
    """Markdown tables of the X2-b report."""
    lines = ["# X2-b on Can MH (development numbers on the 50 X2 validation starts; not a result)",
             "", "Selection by the frozen §7 rule per run; configuration by the frozen §6.3 rule "
             "per family (tuning seed 0). Pilot runs (X2-a, D-124) count as their trials' "
             "seed-0 runs (D-125).", "",
             "| Family | Trial | Seed | Run | Pilot | Status | Wall h | Peak RSS GiB | "
             "Evaluated | Selected step | Successes / 50 | Mean first-success step |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["runs"]:
        t = r["timing"] or {}
        c = (r["selection"] or {}).get("chosen") or {}
        fs = c.get("first_success_step_mean")
        rss = r["peak_rss_bytes"]
        wall = t.get("wall_s")
        lines.append(
            f"| {r['family']} | {r['trial']} | {r['seed']} | `{r['run_id']}` | "
            f"{'yes' if r['pilot'] else ''} | {r['status'] or 'pending'} | "
            f"{'' if wall is None else f'{wall / 3600:.2f}'} | "
            f"{'' if rss is None else f'{rss / 2**30:.2f}'} | {r['n_evaluated']} | "
            f"{c.get('step', '')} | {c.get('successes', '')} | "
            f"{'' if fs is None else f'{fs:.0f}'} |")
    lines += ["", "Chosen configurations: " + (", ".join(
        f"{f} trial {c['chosen']['trial']} ({c['chosen']['successes']}/50)"
        for f, c in report["choices"].items()) or "none yet"), ""]
    if tc is not None:
        lines += ["X2 test configuration (for X2-c): " + "; ".join(
            f"{f}: " + ", ".join(f"seed {r['seed']} step {r['selected_step']} "
                                f"({r['validation_successes']}/50)" for r in v["runs"])
            for f, v in tc["families"].items()), ""]
    e = report["evaluations"]
    lines += [f"X2-b evaluations: {e['x2b_summaries']}, {e['x2b_container_hours']} "
              f"container-hours (mean {e['x2b_mean_wall_s']} s).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    write_report()
    sys.exit(0)
