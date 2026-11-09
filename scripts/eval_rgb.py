"""RGB study: validate checkpoints on the canonical container, then select one per run.

Session 8; RESEARCH_PROTOCOL §7, applied as in Level 1; the RGB freeze.

    # canonical container, one run's checkpoints (all, or --steps); <= 4 containers at once
    make rgb-validate RGB_RUN=<run_id> [RGB_STEPS="5000 10000"]
    # host: the §7 rule over every eligible checkpoint summary of the run
    uv run --frozen python scripts/eval_rgb.py select --run-id <run_id>

Per checkpoint, ``validate`` writes:
* the summary ``reports/session<N>/rgb_validation/<run_id>/step_<k>.json``;
* the episode records ``runs/rgb_eval/<eval_run_id>/``.

``select`` writes ``reports/session<N>/rgb_selection/<run_id>.json``. The rule:
1. the highest validation success count;
2. then the lower mean first-success step;
3. then the earlier checkpoint.

It uses only summaries that are canonical, complete (all 50 starts), verified, and
error-free. It refuses to select until every checkpoint of the run has one.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import sha256_file, write_json  # noqa: E402

RULE = ("highest validation success count; ties: lower mean first-success step over the "
        "successful episodes; remaining ties: earlier checkpoint (RESEARCH_PROTOCOL §7)")


def validate(run_id: str, steps: list[int] | None, n_states: int | None) -> int:
    """Evaluate the run's checkpoints (all by default) on the validation suite."""
    from roboforge.eval.rgb import evaluate_rgb_checkpoint

    run_dir = REPO_ROOT / "runs" / run_id
    man = json.loads((run_dir / "manifest.json").read_text())
    todo = steps or [c["step"] for c in man["checkpoints"]]
    out = evidence_reports_dir("rgb_validation", run_id)
    for step in todo:
        dest = out / f"step_{step:07d}.json"
        if dest.exists() and n_states is None:
            print(f"skip {repo_relative(dest)} (exists)")
            continue
        s = evaluate_rgb_checkpoint(run_dir, step, out_root=REPO_ROOT / "runs" / "rgb_eval",
                                    n_states=n_states)
        brief = {k: s[k] for k in ("eval_run_id", "successes", "n_episodes", "n_errors",
                                   "eligible_for_selection", "verify_passed", "wall_s")}
        brief["first_success_mean"] = s["first_success_step"]["mean"]
        if n_states is None:
            write_json(dest, {k: v for k, v in s.items() if k != "episodes"} |
                       {"episodes_sha256": sha256_file(REPO_ROOT / s["records_dir"] /
                                                       "manifest.json")})
        print(json.dumps(brief), flush=True)
    return 0


def select(run_id: str) -> dict[str, Any]:
    """Apply the §7 rule; raises unless every checkpoint has an eligible summary."""
    man = json.loads((REPO_ROOT / "runs" / run_id / "manifest.json").read_text())
    if man["status"] != "complete":
        raise SystemExit(f"{run_id}: training status {man['status']!r}")
    vdir = evidence_reports_dir("rgb_validation", run_id)
    rows = []
    for c in man["checkpoints"]:
        p = vdir / f"step_{c['step']:07d}.json"
        if not p.exists():
            raise SystemExit(f"{run_id}: step {c['step']} has no validation summary")
        s = json.loads(p.read_text())
        ok = (s["eligible_for_selection"] and s["verify_passed"] and s["n_errors"] == 0
              and s["checkpoint"]["sha256"] == c["sha256"])
        if not ok:
            raise SystemExit(f"{repo_relative(p)}: not eligible (canonical, complete, "
                             f"verified, error-free, matching checkpoint hash)")
        rows.append({"step": c["step"], "sha256": c["sha256"], "successes": s["successes"],
                     "first_success_mean": s["first_success_step"]["mean"],
                     "eval_run_id": s["eval_run_id"], "summary": repo_relative(p),
                     "summary_sha256": sha256_file(p)})
    inf = float("inf")
    best = sorted(rows, key=lambda r: (-r["successes"], r["first_success_mean"]
                                       if r["first_success_mean"] is not None else inf,
                                       r["step"]))[0]
    out = {"schema": "roboforge.rgb_selection/1", "run_id": run_id, "family": man["family"],
           "seed": man["seed"], "rule": RULE, "checkpoints": rows, "selected": best,
           "label": "validation numbers are development numbers (max over K biases upward)"}
    write_json(evidence_reports_dir("rgb_selection") / f"{run_id}.json", out)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--run-id", required=True)
    v.add_argument("--steps", type=int, nargs="*")
    v.add_argument("--n-states", type=int, help="first N starts only (smoke; never selects)")
    s = sub.add_parser("select")
    s.add_argument("--run-id", required=True)
    a = p.parse_args(argv)
    if a.cmd == "validate":
        return validate(a.run_id, a.steps or None, a.n_states)
    r = select(a.run_id)
    print(json.dumps({"run_id": r["run_id"], "selected": r["selected"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
