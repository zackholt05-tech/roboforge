"""Session 6 run accounting from the primary records (manifests, selections, suite events).

    uv run --frozen python logs/session6/analysis/run_table.py > logs/session6/analysis/run_table.md

Reads only ``runs/<id>/manifest.json`` (training), ``reports/session6/selection/*.json`` and
``logs/session6/suite/level1_core_lift/events.jsonl``; writes a markdown table of every Session 6
main-suite training run (failed ones included) plus compute totals. Nothing here selects.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PLAN = json.loads((ROOT / "logs/session6/suite/level1_core_lift/plan_resolved.json").read_text())
EVENTS = [json.loads(x) for x in
          (ROOT / "logs/session6/suite/level1_core_lift/events.jsonl").read_text().splitlines()]


def _runs() -> list[dict]:
    """Every planned run (tuning + replicate) that the suite launched, in event order."""
    launched = []
    for e in EVENTS:
        if e.get("event") == "launch" and e.get("kind") == "train":
            rid = e["key"]
            if rid not in launched:
                launched.append(rid)
    return launched


def main() -> None:
    """Print the table and totals."""
    rows, tot_wall, tot_cpu = [], 0.0, 0.0
    for rid in _runs():
        man_p = ROOT / "runs" / rid / "manifest.json"
        man = json.loads(man_p.read_text()) if man_p.exists() else {}
        sel_p = ROOT / "reports/session6/selection" / f"{rid}.json"
        sel = json.loads(sel_p.read_text())["chosen"] if sel_p.exists() else None
        t = man.get("timing", {})
        tot_wall += t.get("wall_s", 0.0)
        tot_cpu += t.get("cpu_s", 0.0)
        cfg = man.get("config", {})
        git = man.get("git", {})
        cells = [f"`{rid}`", str(cfg.get("seed", "?")), man.get("status", "running/missing"),
                 f"{man['param_count']:,}" if man else "?",
                 f"{t.get('wall_s', 0) / 3600:.2f}", f"{t.get('cpu_s', 0) / 3600:.2f}",
                 str(t.get("n_segments", "?")),
                 git.get("sha", "?")[:7] + ("-dirty" if git.get("dirty") else "")]
        if sel:
            cells += [str(sel["step"]), f"{sel['successes']}/{sel['n_episodes']}",
                      f"{sel['first_success_step_mean']:.2f}",
                      f"[{sel['wilson95'][0]:.3f}, {sel['wilson95'][1]:.3f}]",
                      f"`{sel['sha256'][:16]}`"]
        else:
            cells += ["-"] * 5
        rows.append("| " + " | ".join(cells) + " |")
    ev_wall = sum(e.get("wall_s", 0.0) for e in EVENTS
                  if e.get("event") == "finished" and e.get("kind") == "eval")
    n_eval = sum(1 for e in EVENTS if e.get("event") == "finished" and e.get("kind") == "eval")
    n_eval_fail = sum(1 for e in EVENTS if e.get("event") == "finished" and e.get("kind") == "eval"
                      and e.get("returncode") != 0)
    n_train_fail = sum(1 for e in EVENTS if e.get("event") == "finished" and e.get("kind") == "train"
                       and e.get("returncode") != 0)
    print("| run | seed | status | params | train wall h | train CPU h | segments | git | "
          "selected step | successes | mean first-success step | Wilson 95 % | selected sha256 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print()
    print(f"- training runs: {len(rows)}; non-zero training exits: {n_train_fail}")
    print(f"- training wall {tot_wall / 3600:.2f} h, CPU {tot_cpu / 3600:.2f} core-h (1 torch thread)")
    print(f"- checkpoint evaluations: {n_eval} (non-zero exits {n_eval_fail}); container wall "
          f"{ev_wall / 3600:.2f} container-h (one container per checkpoint, 50 episodes each)")
    first = min(e["utc"] for e in EVENTS if "utc" in e)
    last = max(e["utc"] for e in EVENTS if "utc" in e)
    print(f"- suite events span {first} .. {last}; plan sha256 {PLAN.get('plan_sha256', '?')}")


if __name__ == "__main__":
    main()
