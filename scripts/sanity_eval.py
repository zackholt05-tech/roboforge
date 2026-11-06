"""Closed-loop sanity baselines (random, no-op) on an evaluation suite (engineering control).

Runs every start of ``--suite`` (default ``validation``; the final-test suite raises
``FinalTestLockedError`` until ``docs/FINAL_TEST_FREEZE.md`` exists) through RoboForgeEnv and
the recorder:

* full per-episode records → ``runs/<run_id>/episodes/<state_id>/`` (gitignored);
* run manifest → ``runs/<run_id>/manifest.json``;
* summary (numerator/denominator, stop reasons, errors, clip counts, throughput, a re-simulation
  audit of the first ``--verify`` episodes) → ``reports/session2/sanity_eval/<run_id>.json``.

These are non-learned sanity baselines, not research results: MASTER_SPEC §3.2 item 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import numpy as np  # noqa: E402

from roboforge.eval.policies import NoOpPolicy, RandomPolicy  # noqa: E402
from roboforge.eval.rollout import run_episode, verify_record  # noqa: E402
from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import environment_snapshot, evaluation_platform, write_json  # noqa: E402,E501
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.eval_states import load_suite, read_manifest  # noqa: E402

REPORTS = REPO_ROOT / "reports" / "session2" / "sanity_eval"
RUNS = REPO_ROOT / "runs"


def make_policy(name: str, env: RoboForgeEnv, seed: int) -> Any:
    """Construct a sanity policy by name."""
    if name == "random":
        return RandomPolicy(env.action_spec, seed)
    if name == "noop":
        return NoOpPolicy(env.action_spec)
    raise ValueError(f"unknown sanity policy {name!r}")


def run(task: str, policy_name: str, suite: str, seed: int, n_verify: int,
        runs_dir: Path) -> dict[str, Any]:
    """Evaluate one sanity policy on one suite; return the summary."""
    states = load_suite(task, suite)
    env = RoboForgeEnv(task)
    policy = make_policy(policy_name, env, seed)
    platform = evaluation_platform()
    run_id = (f"s2_sanity_{policy.info.name}_seed{seed}_{task}_{suite}_"
              f"{platform['platform_id']}")
    run_dir = runs_dir / run_id
    episodes, verified = [], []
    t0 = time.perf_counter()
    for i, init in enumerate(states):
        te = time.perf_counter()
        rec = run_episode(env, policy, init, run_id=run_id, episode_index=i)
        wall = time.perf_counter() - te
        rec.save(run_dir / "episodes" / init.state_id)
        if i < n_verify:
            verified.append({"state_id": init.state_id, **verify_record(rec, init)})
        episodes.append({"state_id": init.state_id, "partition": init.partition,
                         "physics_sha256": rec.meta["physics_sha256"], "wall_s": wall,
                         **{k: rec.outcome[k] for k in ("success", "first_success_step",
                                                        "steps", "terminated", "truncated",
                                                        "stop_reason", "error",
                                                        "n_actions_with_clip")}})
    total = time.perf_counter() - t0
    env.close()
    steps = np.array([e["steps"] for e in episodes])
    summary = {
        "run_id": run_id, "kind": "engineering control (non-learned sanity baseline)",
        "task": task, "suite": suite,
        "suite_manifest_content_sha256": read_manifest(task, suite)["content_sha256"],
        "policy": policy.info.to_dict(), "platform": platform,
        "environment": environment_snapshot(),
        "episode_rules": {"horizon": env.horizon, "termination": env.termination,
                          "hold_steps": env.hold_steps},
        "successes": int(sum(e["success"] for e in episodes)), "n_episodes": len(episodes),
        "stop_reasons": {r: sum(e["stop_reason"] == r for e in episodes)
                         for r in sorted({e["stop_reason"] for e in episodes})},
        "n_errors": sum(e["error"] is not None for e in episodes),
        "n_episodes_with_clipping": sum(e["n_actions_with_clip"] > 0 for e in episodes),
        "steps": {"min": int(steps.min()), "max": int(steps.max()), "total": int(steps.sum())},
        "throughput_control_steps_per_s": float(steps.sum() / total),
        "wall_s_total": total, "verify_record": verified,
        "records_dir": repo_relative(run_dir), "episodes": episodes}
    write_json(run_dir / "manifest.json", {k: v for k, v in summary.items() if k != "episodes"})
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=["lift", "can"])
    p.add_argument("--policies", nargs="+", default=["random", "noop"])
    p.add_argument("--suite", default="validation")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--verify", type=int, default=3, help="re-simulate the first N episodes")
    p.add_argument("--out-dir", type=Path, default=REPORTS)
    p.add_argument("--runs-dir", type=Path, default=RUNS)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for task in args.tasks:
        for pol in args.policies:
            s = run(task, pol, args.suite, args.policy_seed, args.verify, args.runs_dir)
            (args.out_dir / f"{s['run_id']}.json").write_text(json.dumps(s, indent=1) + "\n")
            good = s["n_errors"] == 0 and all(v["passed"] for v in s["verify_record"])
            ok &= good
            print(f"[{task}/{pol}] success {s['successes']}/{s['n_episodes']}; stop "
                  f"{s['stop_reasons']}; errors {s['n_errors']}; "
                  f"{s['throughput_control_steps_per_s']:.0f} steps/s; re-sim audit "
                  f"{'PASS' if good else 'FAIL'} -> {s['run_id']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
