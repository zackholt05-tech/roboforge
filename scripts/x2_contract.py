"""X2: show that the validated Can contract applies to the Can MH data (DECISIONS D-121).

Writes ``reports/session<N>/x2/contract.json`` (:mod:`roboforge.x2.contract`): env_args of both
MH files vs ``configs/simulator/can.yaml``; X2 spec hashes vs the committed Can PH specs; the
X2 task spec vs Can PH's; ``RoboForgeEnv('can')`` outside and inside the X2 binding; and the
open-loop replay of every MH demo (engineering check, reported not gated, KI-004)::

    ROBOFORGE_EVIDENCE_SESSION=15 python scripts/x2_contract.py
"""

from __future__ import annotations

import argparse
import json
import sys

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.data import pipeline  # noqa: E402
from roboforge.paths import evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import environment_snapshot, git_revision, write_json  # noqa: E402
from roboforge.x2 import contract as c  # noqa: E402
from roboforge.x2.data import load_x2_artifacts, source_section  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--no-replay", action="store_true", help="skip the open-loop replay")
    args = p.parse_args(argv)
    _, paths, shas = source_section()
    x2 = load_x2_artifacts()["task_manifest"]
    ph = pipeline.load_task_artifacts("can")["task_manifest"]
    specs = {k: {"x2": x2[k], "can_ph": ph[k], "equal": x2[k] == ph[k]}
             for k in ("observation_spec_sha256", "action_spec_sha256")}
    operators = {e["id"]: e["operator"] for e in x2["episodes"]}
    rep = {"schema": "roboforge.x2_contract/1", "decision": "D-121",
           "source_sha256": shas, "x2_task_manifest_sha256": x2["content_sha256"],
           "env_args": c.env_args_check(paths), "spec_hashes": specs,
           "task_spec": c.task_spec_diff(x2["task_spec"], ph["task_spec"]),
           "state_dim": {"x2": x2["task_spec"]["state_dim"],
                         "can_ph": ph["task_spec"]["state_dim"]},
           "features_vs_dataset": x2["verification"],
           "bound_env": c.bound_env_check(),
           "git": git_revision(), "environment": environment_snapshot()}
    if not args.no_replay:
        rep["replay"] = c.replay_all(paths["low_dim"], operators)
    checks = {
        "env_args_task_kwargs_equal": all(v["task_kwargs_equal"] and v["env_name_equal"]
                                          and v["type_equal"] for v in rep["env_args"].values()),
        "spec_hashes_equal_can_ph": all(v["equal"] for v in specs.values()),
        "task_spec_only_allowed_diffs": rep["task_spec"]["only_allowed_fields_differ"],
        "state_dim_equal": x2["task_spec"]["state_dim"] == ph["task_spec"]["state_dim"],
        "features_exact": x2["verification"]["all_equal_features_within_atol"],
        "all_saved_final_states_successful": x2["verification"]["final_state_success"]
        == len(x2["episodes"]),
        "bound_env_horizon_500": rep["bound_env"]["inside"]["horizon"] == 500,
        "bound_env_same_specs": rep["bound_env"]["same_obs_action_specs"],
    }
    if "replay" in rep:
        checks["replay_restore_exact"] = rep["replay"]["max_reset_state_error"] == 0.0
    rep["checks"] = checks
    rep["passed"] = all(checks.values())
    out = evidence_reports_dir("x2") / "contract.json"
    write_json(out, rep)
    summary = {k: v for k, v in rep.items() if k not in ("environment", "replay", "bound_env")}
    if "replay" in rep:
        summary["replay"] = {k: v for k, v in rep["replay"].items() if k != "per_demo"}
    print(json.dumps({**summary, "out": repo_relative(out)}, indent=1))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
