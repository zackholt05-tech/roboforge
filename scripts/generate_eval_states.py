"""Generate (or ``--check``) the evaluation initial-state suites (DECISIONS D-025).

Writes, per task, ``data/eval_states/<task>/{validation,final_test}.json`` (open manifests),
``*.data.jsonl.xz`` (snapshots) and ``partitions.json`` (index of every partition). It also
writes ``reports/session2/eval_states_<task>.json``, which compares the **validation** starts
with the demonstrations' initial states. Only counts, seeds and hashes of the final-test suite
are reported.

``--check`` regenerates everything in memory and fails on any hash difference; it writes
nothing and never reads the locked final-test data file.

Usage::

    python scripts/generate_eval_states.py --tasks lift can          # first generation
    python scripts/generate_eval_states.py --tasks lift can --check  # reproducibility check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from roboforge.data.fetch import load_manifest  # noqa: E402
from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import environment_snapshot, write_json  # noqa: E402
from roboforge.sim import eval_states as es  # noqa: E402
from roboforge.sim.factory import load_sim_config  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "session2"


def _yaw(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def start_summary(task: str, states: np.ndarray, nq: int) -> dict[str, Any]:
    """Task-object position/yaw and arm-joint statistics of a set of flattened start states."""
    from roboforge.sim.factory import make_env
    from roboforge.sim.state_replay import state_layout

    env = make_env(load_sim_config(task))
    layout = state_layout(env).restricted_to(load_sim_config(task)["task_object_joints"])
    env.close()
    pos = states[:, layout.object_pos_idx[0]]
    yaw = np.array([_yaw(s[layout.object_quat_idx[0]]) for s in states])
    arm = states[:, layout.robot_qpos_idx[:7]]
    return {"n": int(len(states)), "object_pos_min": pos.min(0).tolist(),
            "object_pos_max": pos.max(0).tolist(), "object_yaw_min": float(yaw.min()),
            "object_yaw_max": float(yaw.max()), "arm_qpos_mean": arm.mean(0).tolist(),
            "arm_qpos_std": arm.std(0).tolist(), "_object_xy": pos[:, :2]}


def demo_starts(task: str) -> np.ndarray:
    """``states[0]`` of every demonstration (dataset)."""
    _, files = load_manifest()
    low = next(f for f in files if f.task == task and f.hdf5_type == "low_dim").destination()
    with h5py.File(low, "r") as f:
        return np.stack([f["data"][d]["states"][0] for d in f["data"]])


def report(task: str, arts: dict[str, es.SuiteArtifacts]) -> dict[str, Any]:
    """Validation-vs-demonstration start-distribution report (final test: hashes only)."""
    val = np.stack([s.state for s in arts["validation"].states])
    demo = demo_starts(task)
    nq = (val.shape[1] - 1) // 2
    sv, sd = start_summary(task, val, nq), start_summary(task, demo, nq)
    dist = np.linalg.norm(sv.pop("_object_xy")[:, None, :] - sd.pop("_object_xy")[None], axis=2)
    nearest = dist.min(axis=1)
    test = arts["final_test"].manifest
    return {"task": task, "environment": environment_snapshot(),
            "validation": {"manifest_content_sha256": arts["validation"].manifest["content_sha256"],
                           "starts": sv, "n_unique_models":
                               arts["validation"].manifest["n_unique_models"]},
            "demonstration_starts": sd,
            "validation_to_nearest_demo_start_object_xy_m": {
                "min": float(nearest.min()), "median": float(np.median(nearest)),
                "max": float(nearest.max())},
            "final_test": {"n_states": test["n_states"], "seed_base": test["seed_base"],
                           "manifest_content_sha256": test["content_sha256"],
                           "details": "withheld until docs/FINAL_TEST_FREEZE.md"},
            "suites_disjoint": not ({s.state_sha256 for s in arts["validation"].states}
                                    & {s.state_sha256 for s in arts["final_test"].states})}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=["lift", "can"])
    p.add_argument("--check", action="store_true", help="regenerate in memory and compare")
    p.add_argument("--overwrite", action="store_true",
                   help="replace committed suites (only for a recorded decision)")
    p.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = p.parse_args(argv)
    ok = True
    for task in args.tasks:
        if args.check:
            problems, details = es.check_task(task)
            idx = es.partitions_index(task)
            committed = json.loads((es.suite_dir(task) / "partitions.json").read_text())
            if committed != idx:
                problems.append(f"{task}: partitions.json differs from the rebuilt index")
            for name in es.suites_config()["suites"]:
                if not es.verify_data_file(task, name):
                    problems.append(f"{task}/{name}: data file does not match its manifest")
            print(f"[{task}] check: {'PASS' if not problems else 'FAIL'} {json.dumps(details)}")
            for prob in problems:
                print("   ", prob)
            ok &= not problems
            continue
        arts = {name: es.generate_suite(task, name) for name in es.suites_config()["suites"]}
        for name, art in arts.items():
            paths = es.write_suite(art, overwrite=args.overwrite)
            print(f"[{task}/{name}] {art.manifest['n_states']} states, "
                  f"{art.manifest['n_unique_models']} unique models, content "
                  f"{art.manifest['content_sha256']} -> {paths}")
        write_json(es.suite_dir(task) / "partitions.json", es.partitions_index(task))
        rep = report(task, arts)
        args.report_dir.mkdir(parents=True, exist_ok=True)
        out = args.report_dir / f"eval_states_{task}.json"
        out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
        print(f"[{task}] report {repo_relative(out)}; suites disjoint: {rep['suites_disjoint']}")
        ok &= rep["suites_disjoint"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
