"""Action-convention diagnostics for trained BC checkpoints (MASTER_SPEC Session 3).

    python scripts/diagnose_actions.py --train-runs runs/<id1> runs/<id2>

Looks for clipping, sign and coordinate problems from three directions:

1. **Controller axes** (simulator, validation start 0). Command one action dimension at a
   time (+0.5 for 4 control steps, everything else 0), then check:
   - each Δposition dimension moves the end effector mostly along the same **world** axis,
     with a positive sign;
   - each Δrotation dimension rotates it mostly about that world axis, positively
     (OSC_POSE delta, ``input_ref_frame=world``);
   - gripper +1 closes the fingers and −1 opens them.

   Positions and orientations come from the canonical features (eef site position, hand
   quaternion xyzw), and the rotation increment is computed in float64 with numpy, not with
   robosuite helpers (KI-018).
2. **Teacher-forced offline error** on the 20 validation demos, per action dimension:
   - normalized MSE and RMSE in action units;
   - gripper sign agreement with the demonstration;
   - the fraction of decoded actions that fall outside [−1, 1] and would be clipped.
3. **Closed loop**: per-dimension clip fractions and gripper statistics from the run's
   canonical validation summaries, when they exist.

The report goes to ``reports/session3/diagnostics/actions.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from roboforge.eval.learned import LearnedPolicy  # noqa: E402
from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import evaluation_platform, git_revision  # noqa: E402
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.eval_states import load_suite  # noqa: E402
from roboforge.train.config import from_dict  # noqa: E402
from roboforge.train.run import load_windows, read_manifest  # noqa: E402


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion (x, y, z, w), float64."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec(r: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (angle < pi)."""
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))
    if angle < 1e-12:
        return np.zeros(3)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return angle * v / (2 * np.sin(angle))


def controller_axes(task: str) -> dict[str, Any]:
    """Command each action dimension alone; report the world-frame response."""
    env = RoboForgeEnv(task)
    init = load_suite(task, "validation")[0]
    sl = env.observation_spec.slices()
    pos, quat, grip = (slice(*sl[k]) for k in ("robot0_eef_pos", "robot0_eef_quat",
                                                 "robot0_gripper_qpos"))
    rows = []
    try:
        for dim in range(7):
            for sign in ((1.0, -1.0) if dim == 6 else (1.0,)):
                obs, _ = env.reset(init)
                if dim == 6 and sign < 0:  # close first, then test opening
                    for _ in range(10):
                        obs = env.step(np.eye(7)[6])[0]
                o0 = obs.vector
                a = np.zeros(7)
                a[dim] = 0.5 * sign if dim < 6 else sign
                for _ in range(4 if dim < 6 else 10):
                    obs = env.step(a)[0]
                o1 = obs.vector
                dp = o1[pos] - o0[pos]
                dr = rotvec(quat_xyzw_to_mat(o1[quat]) @ quat_xyzw_to_mat(o0[quat]).T)
                gap0, gap1 = o0[grip][0] - o0[grip][1], o1[grip][0] - o1[grip][1]
                row = {"dim": dim, "name": env.action_spec.names[dim], "command": float(a[dim]),
                       "delta_pos_m": dp.tolist(), "delta_rotvec_rad": dr.tolist(),
                       "finger_gap_before_m": float(gap0), "finger_gap_after_m": float(gap1)}
                if dim < 3:
                    row["ok"] = bool(np.argmax(np.abs(dp)) == dim and dp[dim] > 0)
                elif dim < 6:
                    row["ok"] = bool(np.argmax(np.abs(dr)) == dim - 3 and dr[dim - 3] > 0)
                else:
                    row["ok"] = bool(gap1 < gap0) if sign > 0 else bool(gap1 > gap0)
                rows.append(row)
    finally:
        env.close()
    return {"task": task, "state_id": init.state_id, "rows": rows,
            "passed": all(r["ok"] for r in rows)}


@torch.no_grad()
def teacher_forced(policy: LearnedPolicy, tm: dict[str, Any]) -> dict[str, Any]:
    """Per-dimension offline error on the validation demos, in normalized and action units."""
    _, valid = load_windows(from_dict(tm["config"]))
    pred = policy.model(valid.obs_history, valid.obs_valid).double().numpy()
    target = valid.future_action.double().numpy()
    m = valid.valid_mask.numpy()
    p, y = pred[m], target[m]  # [N_valid_steps, A] normalized
    dp, dy = (policy.action_normalizer.unnormalize(p), policy.action_normalizer.unnormalize(y))
    return {"n_steps": int(len(p)),
            "mse_normalized_per_dim": ((p - y) ** 2).mean(0).tolist(),
            "rmse_action_units_per_dim": np.sqrt(((dp - dy) ** 2).mean(0)).tolist(),
            "gripper_sign_agreement": float((np.sign(dp[:, -1]) == np.sign(dy[:, -1])).mean()),
            "decoded_outside_bounds_per_dim": (np.abs(dp) > 1.0).mean(0).tolist(),
            "decoded_min_per_dim": dp.min(0).tolist(), "decoded_max_per_dim": dp.max(0).tolist(),
            "demo_min_per_dim": dy.min(0).tolist(), "demo_max_per_dim": dy.max(0).tolist()}


def closed_loop(tm: dict[str, Any], reports: Path) -> list[dict[str, Any]]:
    """Clip and gripper statistics from the run's canonical validation summaries."""
    out = []
    for f in sorted((reports / "validation").glob(f"{tm['run_id']}__val_step*.json")):
        s = json.loads(f.read_text())
        eps = [e for e in s["episodes"] if e["steps"] > 0]
        out.append({"step": s["checkpoint"]["step"], "successes": s["successes"],
                    "n": s["n_episodes"], "clip_fraction_per_dim": s["clip_fraction_per_dim"],
                    "decoded_min_per_dim": np.min([e["decoded_min"] for e in eps], 0).tolist(),
                    "decoded_max_per_dim": np.max([e["decoded_max"] for e in eps], 0).tolist(),
                    "gripper_sign_changes_mean": s["gripper_sign_changes_mean"],
                    "source": repo_relative(f)})
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-runs", type=Path, nargs="+", required=True)
    p.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports" / "session3")
    args = p.parse_args(argv)
    runs, tasks = [], set()
    for run in args.train_runs:
        tm = read_manifest(run)
        tasks.add(tm["config"]["task"])
        last = tm["checkpoints"][-1]
        policy = LearnedPolicy.from_checkpoint(REPO_ROOT / last["path"], last["sha256"])
        runs.append({"train_run_id": tm["run_id"], "name": tm["config"]["name"],
                     "checkpoint": last, "teacher_forced_valid_demos": teacher_forced(policy, tm),
                     "closed_loop_validation": closed_loop(tm, args.reports_dir)})
    axes = {t: controller_axes(t) for t in sorted(tasks)}
    report = {"schema": "roboforge.action_diagnostics/1", "platform": evaluation_platform(),
              "git": git_revision(), "controller_axes": axes, "runs": runs}
    out = args.reports_dir / "diagnostics" / "actions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    for t, a in axes.items():
        print(f"[{t}] controller axes/signs {'PASS' if a['passed'] else 'FAIL'}: " + ", ".join(
            f"{r['name']}{'+' if r['command'] > 0 else '-'}:{'ok' if r['ok'] else 'BAD'}"
            for r in a["rows"]))
    for r in runs:
        tf = r["teacher_forced_valid_demos"]
        print(f"{r['name']}: gripper sign agreement {tf['gripper_sign_agreement']:.3f}; RMSE "
              f"{[round(x, 3) for x in tf['rmse_action_units_per_dim']]}; outside bounds "
              f"{[round(x, 3) for x in tf['decoded_outside_bounds_per_dim']]}")
    print(f"-> {repo_relative(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
