"""Level 2 study: validate and select point-cloud checkpoints, and run the placement evaluation.

Session 9; RESEARCH_PROTOCOL §7, applied as in Level 1 and the RGB study; the Level 2 plan
``configs/experiments/level2_pointcloud.yaml`` (fixed by ``docs/LEVEL2_EXPERIMENT_FREEZE.md``).

    # canonical container: one run's checkpoints (all, or --steps); <= 4 containers at once
    make pc-validate PC_RUN=<run_id> [PC_STEPS="5000 10000"]
    # host: the §7 rule over every eligible checkpoint summary of the run
    uv run --frozen python scripts/eval_pc.py select --run-id <run_id>
    # canonical container, once per (arm, seed, suite), after every selection exists
    make pc-placement PC_ARM=pc|rgb PC_SEED=<s> PC_SUITE=robustness_base|ood_cube_shift

``validate`` writes, per checkpoint:
* the summary ``reports/session<N>/pc_validation/<run_id>/step_<k>.json``;
* the episode records ``runs/pc_eval/<eval_run_id>/``.

``select`` writes ``reports/session<N>/pc_selection/<run_id>.json``. The rule:
1. the highest validation success count;
2. then the lower mean first-success step;
3. then the earlier checkpoint.

It uses only canonical, complete, verified, error-free summaries whose checkpoint hash matches
the training manifest. It refuses until every checkpoint has one.

``placement`` evaluates one arm's selected checkpoint once on a preregistered placement suite:
* the point-cloud arm's selection is read from its committed selection artifact;
* the frozen RGB diffusion arm's selection is read from the plan, where it is pinned by hash.

It writes ``reports/session<N>/level2_placement/<arm>/<run_id>__<suite>.json`` and never
overwrites it. Nothing here can reach the held-out suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import yaml  # noqa: E402

from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import sha256_file, write_json  # noqa: E402

RULE = ("highest validation success count; ties: lower mean first-success step over the "
        "successful episodes; remaining ties: earlier checkpoint (RESEARCH_PROTOCOL §7)")
PLAN = REPO_ROOT / "configs" / "experiments" / "level2_pointcloud.yaml"
PLACEMENT_SUITES = ("robustness_base", "ood_cube_shift")


def plan() -> dict[str, Any]:
    """The committed Level 2 plan."""
    return yaml.safe_load(PLAN.read_text())


def _brief(s: dict[str, Any]) -> dict[str, Any]:
    b = {k: s[k] for k in ("eval_run_id", "successes", "n_episodes", "n_errors",
                           "verify_passed", "wall_s")}
    b["first_success_mean"] = s["first_success_step"]["mean"]
    return b


def _write_summary(dest: Path, s: dict[str, Any]) -> None:
    write_json(dest, {k: v for k, v in s.items() if k != "episodes"} |
               {"episodes_sha256": sha256_file(REPO_ROOT / s["records_dir"] / "manifest.json")})


def validate(run_id: str, steps: list[int] | None, n_states: int | None) -> int:
    """Evaluate the run's checkpoints (all by default) on the validation suite."""
    from roboforge.eval.pointcloud import evaluate

    run_dir = REPO_ROOT / "runs" / run_id
    man = json.loads((run_dir / "manifest.json").read_text())
    todo = steps or [c["step"] for c in man["checkpoints"]]
    out = evidence_reports_dir("pc_validation", run_id)
    for step in todo:
        dest = out / f"step_{step:07d}.json"
        if dest.exists() and n_states is None:
            print(f"skip {repo_relative(dest)} (exists)")
            continue
        s = evaluate("pc", run_dir, step, suite="validation",
                     out_root=REPO_ROOT / "runs" / "pc_eval", n_states=n_states)
        if n_states is None:
            _write_summary(dest, s)
        print(json.dumps(_brief(s) | {"eligible_for_selection": s["eligible_for_selection"]}),
              flush=True)
    return 0


def select(run_id: str) -> dict[str, Any]:
    """Apply the §7 rule; raises unless every checkpoint has an eligible summary."""
    man = json.loads((REPO_ROOT / "runs" / run_id / "manifest.json").read_text())
    if man["status"] != "complete":
        raise SystemExit(f"{run_id}: training status {man['status']!r}")
    vdir = evidence_reports_dir("pc_validation", run_id)
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
    out = {"schema": "roboforge.pc_selection/1", "run_id": run_id, "family": man["family"],
           "seed": man["seed"], "rule": RULE, "checkpoints": rows, "selected": best,
           "label": "validation numbers are development numbers (max over K biases upward)"}
    write_json(evidence_reports_dir("pc_selection") / f"{run_id}.json", out)
    return out


def selected(arm: str, seed: int) -> dict[str, Any]:
    """``{run_id, step, checkpoint_sha256}`` of an arm's selected checkpoint at ``seed``."""
    p = plan()
    if arm == "rgb":
        row = next(r for r in p["arms"]["rgb_diffusion"]["selections"] if r["seed"] == seed)
        return {"run_id": row["run_id"], "step": row["step"],
                "checkpoint_sha256": row["checkpoint_sha256"]}
    from roboforge.train.pointcloud import PCConfig, config_path

    rid = PCConfig.load(config_path(), {"seed": seed}).run_id(p["arms"]["pc_diffusion"]["prefix"])
    sel = json.loads((evidence_reports_dir("pc_selection") / f"{rid}.json").read_text())
    return {"run_id": rid, "step": sel["selected"]["step"],
            "checkpoint_sha256": sel["selected"]["sha256"]}


def placement(arm: str, seed: int, suite: str) -> dict[str, Any]:
    """One arm's selected checkpoint on one preregistered placement suite, once."""
    from roboforge.eval.pointcloud import evaluate

    if suite not in PLACEMENT_SUITES:
        raise SystemExit(f"placement suites are {PLACEMENT_SUITES}")
    sel = selected(arm, seed)
    dest = evidence_reports_dir("level2_placement", arm) / f"{sel['run_id']}__{suite}.json"
    if dest.exists():
        raise SystemExit(f"{repo_relative(dest)} exists: each placement evaluation runs once")
    s = evaluate(arm, REPO_ROOT / "runs" / sel["run_id"], sel["step"], suite=suite,
                 out_root=REPO_ROOT / "runs" / "level2_eval",
                 expected_sha256=sel["checkpoint_sha256"],
                 label=" — preregistered placement evaluation, run once")
    if not s["canonical_platform"]:
        raise SystemExit("placement evaluations count only on the canonical platform")
    _write_summary(dest, s)
    return s


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
    pl = sub.add_parser("placement")
    pl.add_argument("--arm", choices=["pc", "rgb"], required=True)
    pl.add_argument("--seed", type=int, required=True)
    pl.add_argument("--suite", choices=PLACEMENT_SUITES, required=True)
    a = p.parse_args(argv)
    if a.cmd == "validate":
        return validate(a.run_id, a.steps or None, a.n_states)
    if a.cmd == "placement":
        print(json.dumps(_brief(placement(a.arm, a.seed, a.suite))), flush=True)
        return 0
    r = select(a.run_id)
    print(json.dumps({"run_id": r["run_id"], "selected": r["selected"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
