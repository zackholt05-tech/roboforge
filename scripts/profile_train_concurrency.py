"""Local training throughput under concurrency (Session 5, engineering profile).

    python scripts/profile_train_concurrency.py --levels 1 4 6 8 --steps 300 \
        --out reports/session5/cost/train_concurrency.json

Starts K identical one-thread training profiles (:mod:`roboforge.compute.profile`, diffusion by
default, the most expensive family) at the same time and records each process's median
seconds per optimizer step. ``slowdown_at_k`` is the median step time at K divided by the
median at K = 1. ``recommended_k`` is the K with the highest aggregate steps per second. No run
directory or checkpoint is written.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import environment_snapshot, git_revision


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 6, 8])
    ap.add_argument("--family", default="diffusion_unet")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    raw = a.out.parent / "train_concurrency_raw"
    levels = {}
    for k in a.levels:
        outs = [raw / f"k{k}_{i}.json" for i in range(k)]
        procs = [subprocess.Popen([sys.executable, "-m", "roboforge.compute.profile",
                                   "--family", a.family, "--steps", str(a.steps), "--tag",
                                   f"k{k}_{i}", "--out", str(o)], cwd=REPO_ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                 for i, o in enumerate(outs)]
        codes = [p.wait() for p in procs]
        rows = [json.loads(o.read_text()) for o in outs]
        med = sorted(r["median_s"] for r in rows)[len(rows) // 2]
        levels[k] = {"returncodes": codes, "median_step_s": med,
                     "per_process_steps_per_s": [r["steps_per_s_median"] for r in rows],
                     "aggregate_steps_per_s": sum(r["steps_per_s_median"] for r in rows),
                     "raw": [repo_relative(o) for o in outs]}
    base = levels[min(levels)]["median_step_s"]
    slow = {str(k): v["median_step_s"] / base for k, v in levels.items()}
    best = max(levels, key=lambda k: levels[k]["aggregate_steps_per_s"])
    res = {"schema": "roboforge.train_concurrency/1",
           "kind": "engineering profile (local CPU, 1 torch thread per process)",
           "family": a.family, "steps": a.steps, "levels": levels, "slowdown_at_k": slow,
           "recommended_k": best, "git": git_revision(), "environment": environment_snapshot()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"slowdown_at_k": slow, "recommended_k": best,
                      "aggregate": {k: v["aggregate_steps_per_s"] for k, v in levels.items()}},
                     indent=1))
    return 0 if all(c == 0 for v in levels.values() for c in v["returncodes"]) else 1


if __name__ == "__main__":
    sys.exit(main())
