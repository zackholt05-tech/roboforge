"""Probe whether relative object-gripper observation features are physically correct.

Session 0 found that robosuite v1.5.1 PickPlace computes ``Can_to_robot0_eef_pos`` /
``Can_to_robot0_eef_quat`` (the first 7 dims of robomimic's 14-D Can ``object`` obs)
from *cached* object poses of the previous observation update, because the relative
sensor is evaluated before ``Can_pos``/``Can_quat`` in the observable order. Lift's
``gripper_to_cube_pos`` is evaluated after ``cube_pos`` and is fresh.

For each task this script measures:

* ``stored_vs_truth`` – stored dataset relative feature vs. the same feature recomputed
  from the *stored* world-frame object and end-effector poses (all timesteps, N demos);
* ``closed_loop_vs_truth`` – ``env.step`` observation vs. the recomputed truth while
  replaying one demonstration's actions (what a policy would see at evaluation time);
* ``lag_hypothesis_t_gt0`` (Can) – stored relative pose at t vs. the pose built from the
  can pose at t-1 and the end-effector pose at t (≈0 confirms the one-update lag).

Writes ``reports/feasibility/obs_consistency.json``.
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
import robosuite.utils.transform_utils as T  # noqa: E402

from roboforge.data.fetch import load_manifest  # noqa: E402
from roboforge.paths import feasibility_reports_dir  # noqa: E402
from roboforge.provenance import environment_snapshot  # noqa: E402
from roboforge.sim.factory import load_sim_config, make_env  # noqa: E402
from roboforge.sim.state_replay import reset_to_state  # noqa: E402


def can_rel_pose(eef_pos, eef_quat, obj_pos, obj_quat) -> tuple[np.ndarray, np.ndarray]:
    """Can pose expressed in the gripper frame (robosuite convention, xyzw quats)."""
    world_in_grip = T.pose_inv(T.pose2mat((eef_pos, eef_quat)))
    return T.mat2pose(T.pose_in_A_to_pose_in_B(T.pose2mat((obj_pos, obj_quat)), world_in_grip))


def can_rel_pos(eef_pos, eef_quat, obj_pos, obj_quat) -> np.ndarray:
    """Can position expressed in the gripper frame."""
    return can_rel_pose(eef_pos, eef_quat, obj_pos, obj_quat)[0]


def quat_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation angle between unit quaternions (sign-invariant)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(2 * np.arccos(min(1.0, abs(float(np.dot(a, b))))))


def can_lag_hypothesis(f: h5py.File, demos: list[str]) -> dict[str, Any]:
    """Test: stored rel feature at t == rel(can pose at t-1, eef pose at t) for t > 0."""
    res: dict[str, list[float]] = {"pos_same_step": [], "pos_can_tm1_eef_t": [],
                                   "quat_same_step_rad": [], "quat_can_tm1_eef_t_rad": []}
    for d in demos:
        o = f["data"][d]["obs"]
        ep, eq, ob = o["robot0_eef_pos"][()], o["robot0_eef_quat"][()], o["object"][()]
        for t in range(1, len(ob)):
            p_s, q_s = can_rel_pose(ep[t], eq[t], ob[t, 7:10], ob[t, 10:14])
            p_l, q_l = can_rel_pose(ep[t], eq[t], ob[t - 1, 7:10], ob[t - 1, 10:14])
            res["pos_same_step"].append(float(np.abs(ob[t, :3] - p_s).max()))
            res["pos_can_tm1_eef_t"].append(float(np.abs(ob[t, :3] - p_l).max()))
            res["quat_same_step_rad"].append(quat_dist(ob[t, 3:7], q_s))
            res["quat_can_tm1_eef_t_rad"].append(quat_dist(ob[t, 3:7], q_l))
    return {k: {"median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)), "max": max(v)}
            for k, v in res.items()}


def truth_from_obs(task: str, eef_pos, eef_quat, obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (relative feature as stored, recomputed truth) from one object-obs vector."""
    if task == "lift":  # object = [cube_pos(3), cube_quat(4), gripper_to_cube_pos(3)]
        return obj[7:10], obj[0:3] - eef_pos
    # can: object = [Can_to_eef_pos(3), Can_to_eef_quat(4), Can_pos(3), Can_quat(4)]
    return obj[0:3], can_rel_pos(eef_pos, eef_quat, obj[7:10], obj[10:14])


def probe(task: str, n_demos: int) -> dict[str, Any]:
    """Run both consistency measurements for one task."""
    _, files = load_manifest()
    path = next(f for f in files if f.task == task and f.hdf5_type == "low_dim").destination()
    out: dict[str, Any] = {"task": task}
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:n_demos]
        errs, errs_t0, errs_rest = [], [], []
        for d in demos:
            o = f["data"][d]["obs"]
            for t in range(len(o["object"])):
                eef_p, eef_q = o["robot0_eef_pos"][t], o["robot0_eef_quat"][t]
                stored, truth = truth_from_obs(task, eef_p, eef_q,
                                               o["object"][t])
                e = float(np.abs(stored - truth).max())
                errs.append(e)
                (errs_t0 if t == 0 else errs_rest).append(e)
        if task == "can":
            out["lag_hypothesis_t_gt0"] = can_lag_hypothesis(f, demos)
        out["stored_vs_truth_m"] = {
            "n_demos": len(demos), "n_steps": len(errs),
            "max": max(errs), "median": float(np.median(errs)),
            "t0_max": max(errs_t0), "t_gt0_max": max(errs_rest),
            "t_gt0_p95": float(np.quantile(errs_rest, 0.95)),
        }
        g = f["data"][demos[0]]
        env = make_env(load_sim_config(task))
        reset_to_state(env, g.attrs["model_file"], g["states"][0])
        cl = []
        for a in g["actions"][()]:
            ob = env.step(a)[0]
            stored, truth = truth_from_obs(task, ob["robot0_eef_pos"], ob["robot0_eef_quat"],
                                           ob["object-state"])
            cl.append(float(np.abs(stored - truth).max()))
        env.close()
        out["closed_loop_vs_truth_m"] = {
            "demo": demos[0], "max": max(cl), "median": float(np.median(cl))
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=["lift", "can"])
    p.add_argument("--demos", type=int, default=20)
    p.add_argument("--out", type=Path, default=feasibility_reports_dir() / "obs_consistency.json")
    args = p.parse_args(argv)
    results = [probe(t, args.demos) for t in args.tasks]
    report = {"environment": environment_snapshot(), "results": results}
    args.out.write_text(json.dumps(report, indent=1))
    for r in report["results"]:
        s, c = r["stored_vs_truth_m"], r["closed_loop_vs_truth_m"]
        print(f"[{r['task']}] stored-vs-truth max {s['max']:.4f} m (t=0 max {s['t0_max']:.4f}, "
              f"t>0 p95 {s['t_gt0_p95']:.4f}) | closed-loop-vs-truth max {c['max']:.4f} m")
        if "lag_hypothesis_t_gt0" in r:
            lag = r["lag_hypothesis_t_gt0"]
            pos_max = lag["pos_can_tm1_eef_t"]["max"]
            print(f"   lag test: stored == rel(can[t-1], eef[t]) -> pos max {pos_max:.1e} m, "
                  f"quat max {lag['quat_can_tm1_eef_t_rad']['max']:.1e} rad "
                  f"(same-step truth pos p95 {lag['pos_same_step']['p95']:.4f} m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
