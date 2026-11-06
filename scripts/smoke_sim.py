"""Simulator smoke test: real 3D robot reset/step/render without any dataset.

Checks (MASTER_SPEC §2.3 items 3, 5, 7; Session 0 objective C):

1. build the task env from ``configs/simulator/<task>.yaml`` (dataset controller settings);
2. seeded reset: same global seed -> identical initial MuJoCo state. robosuite v1.5.1
   task envs reject a ``seed=`` kwarg; placement uses the global ``np.random`` state;
3. observation dict keys/shapes, action space bounds and dimension;
4. closed-loop random actions: sim time advances by the control timestep, robot moves,
   rewards/success are reported, all values finite;
5. success predicate responds: lifting the cube via a state edit flips ``_check_success``;
6. horizon/termination semantics with ``ignore_done=False``;
7. offscreen RGB render from a named camera, saved as PNG;
8. two fresh Python processes restoring one serialized state and applying the same
   actions produce matching trajectories (max abs state difference reported).

Writes ``reports/feasibility/smoke_sim_<task>.json``. Exit code 1 on any failed check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

BACKEND = configure_mujoco_gl()

import numpy as np  # noqa: E402

from roboforge.paths import feasibility_reports_dir  # noqa: E402
from roboforge.provenance import environment_snapshot  # noqa: E402
from roboforge.sim.factory import load_sim_config, make_env, task_kwargs  # noqa: E402
from roboforge.sim.renderer import render_rgb  # noqa: E402
from roboforge.sim.state_replay import is_success, reset_to_state, state_layout  # noqa: E402


def seeded_reset(env: Any, seed: int) -> np.ndarray:
    """Reset with the global numpy RNG seeded (robosuite's placement uses np.random)."""
    np.random.seed(seed)
    env.reset()
    return np.array(env.sim.get_state().flatten())


def fixed_actions(n: int, dim: int, seed: int) -> np.ndarray:
    """Deterministic uniform random actions in [-1, 1]."""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, dim))


def child_rollout(task: str, state_file: Path) -> None:
    """Child-process mode: restore a serialized state, apply fixed actions, print states."""
    payload = json.loads(state_file.read_text())
    env = make_env(load_sim_config(task))
    reset_to_state(env, payload["model_xml"], np.array(payload["state"]))
    traj = []
    for a in np.array(payload["actions"]):
        env.step(a)
        traj.append(env.sim.get_state().flatten().tolist())
    print(json.dumps({"trajectory": traj}))


def run(task: str, seed: int, n_steps: int, out_dir: Path) -> dict[str, Any]:
    """Execute all smoke checks and return the report."""
    cfg = load_sim_config(task)
    checks: dict[str, bool] = {}
    rep: dict[str, Any] = {"task": task, "seed": seed, "backend": BACKEND,
                           "environment": environment_snapshot()}

    t0 = time.perf_counter()
    env = make_env(cfg, offscreen=True)
    rep["construct_seconds"] = time.perf_counter() - t0

    # 2. reset determinism
    s_a, s_b, s_c = seeded_reset(env, seed), seeded_reset(env, seed), seeded_reset(env, seed + 1)
    import robosuite

    try:  # probe robosuite directly (not our factory) for a constructor seed kwarg
        robosuite.make(cfg["env_name"], seed=seed, has_renderer=False, has_offscreen_renderer=False,
                       use_camera_obs=False, **task_kwargs(cfg)).close()
        seed_kwarg_supported = True
    except TypeError:
        seed_kwarg_supported = False
    layout = state_layout(env).restricted_to(cfg["task_object_joints"])
    obj_q = layout.object_pos_idx[0]
    rep["reset_determinism"] = {
        "same_global_seed_max_abs_diff": float(np.abs(s_a - s_b).max()),
        "different_seed_object_xy_diff_m": float(np.abs(s_a[obj_q[:2]] - s_c[obj_q[:2]]).max()),
        "robosuite_seed_kwarg_supported": seed_kwarg_supported,
    }
    det = rep["reset_determinism"]
    checks["same_seed_identical_reset"] = det["same_global_seed_max_abs_diff"] == 0.0
    checks["different_seed_changes_object"] = det["different_seed_object_xy_diff_m"] > 1e-4

    # 3. observation and action spaces
    seeded_reset(env, seed)
    obs = env._get_observations(force_update=True)
    low, high = env.action_spec
    rep["observation_spec"] = {k: list(np.shape(v)) for k, v in obs.items()}
    rep["action_space"] = {"dim": int(env.action_dim), "low": low.tolist(), "high": high.tolist(),
                           "control_freq_hz": env.control_freq,
                           "control_timestep_s": env.control_timestep,
                           "physics_timestep_s": env.model_timestep,
                           "substeps_per_control": round(env.control_timestep / env.model_timestep)}
    checks["action_dim_7"] = env.action_dim == 7
    rep["kinematics_t0"] = {
        "robot_joint_names": list(env.robots[0].robot_joints),
        "joint_qpos": env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].tolist(),
        "eef_pos_world_m": obs["robot0_eef_pos"].tolist(),
        "eef_quat_xyzw": obs["robot0_eef_quat"].tolist(),
        "object_free_joints": layout.object_names,
        "object_pos_world_m": [
            env.sim.data.qpos[[i - 1 for i in idx]].tolist() for idx in layout.object_pos_idx
        ],
    }

    # 4. closed-loop random actions
    actions = fixed_actions(n_steps, env.action_dim, seed)
    q0, time0 = env.sim.data.qpos.copy(), env.sim.data.time
    rewards, successes, finite = [], [], True
    t0 = time.perf_counter()
    for a in actions:
        ob, r, done, info = env.step(a)
        rewards.append(float(r))
        successes.append(is_success(env))
        finite &= all(np.all(np.isfinite(v)) for v in ob.values() if isinstance(v, np.ndarray))
    dt = time.perf_counter() - t0
    rep["random_rollout"] = {
        "n_steps": n_steps, "sim_time_advanced_s": float(env.sim.data.time - time0),
        "expected_sim_time_s": n_steps * env.control_timestep,
        "robot_qpos_change_max_rad": float(np.abs(env.sim.data.qpos[:7] - q0[:7]).max()),
        "reward_sum": float(np.sum(rewards)), "success_any": bool(any(successes)),
        "all_obs_finite": bool(finite), "control_steps_per_sec": n_steps / dt,
        "info_keys": sorted(info.keys()),
    }
    checks["time_advances_correctly"] = abs(rep["random_rollout"]["sim_time_advanced_s"]
                                            - rep["random_rollout"]["expected_sim_time_s"]) < 1e-6
    checks["robot_moves"] = rep["random_rollout"]["robot_qpos_change_max_rad"] > 1e-3
    checks["random_policy_not_successful"] = not rep["random_rollout"]["success_any"]
    checks["obs_finite"] = bool(finite)

    # 5. success predicate responds to a lifted cube (Lift only: single free body)
    if task == "lift":
        seeded_reset(env, seed)
        before = is_success(env)
        qadr = layout.object_pos_idx[0][2] - 1
        env.sim.data.qpos[qadr] += 0.10
        env.sim.forward()
        after = is_success(env)
        rep["success_predicate"] = {"before_lift": before, "after_raising_cube_10cm": after}
        checks["success_predicate_responds"] = (not before) and after

    # 6. horizon / termination semantics
    env_h = make_env(cfg, horizon=5)
    env_h.ignore_done = False
    np.random.seed(seed)
    env_h.reset()
    dones = [bool(env_h.step(np.zeros(env_h.action_dim))[2]) for _ in range(5)]
    env_h.close()
    rep["termination"] = {"horizon": 5, "done_flags": dones}
    checks["done_only_at_horizon"] = dones == [False, False, False, False, True]

    # 7. render
    out_dir.joinpath("frames").mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    seeded_reset(env, seed)
    t0 = time.perf_counter()
    frame = render_rgb(env, "agentview", 256, 256)
    rep["render"] = {"camera": "agentview", "shape": list(frame.shape), "dtype": str(frame.dtype),
                     "ms": 1000 * (time.perf_counter() - t0), "mean": float(frame.mean()),
                     "std": float(frame.std())}
    frame_path = out_dir / "frames" / f"smoke_{task}_seed{seed}_agentview.png"
    iio.imwrite(frame_path, frame)
    rep["render"]["path"] = str(frame_path.relative_to(out_dir.parents[1]))
    checks["render_nontrivial"] = rep["render"]["std"] > 5.0

    # 8. two fresh processes, same serialized state + actions
    seeded_reset(env, seed)
    payload = {"model_xml": env.sim.model.get_xml(),
               "state": env.sim.get_state().flatten().tolist(),
               "actions": fixed_actions(20, env.action_dim, seed + 7).tolist()}
    env.close()
    state_file = out_dir / f".smoke_state_{task}.json"
    state_file.write_text(json.dumps(payload))
    trajs = []
    for _ in range(2):
        cmd = [sys.executable, __file__, "--task", task, "--child", str(state_file)]
        proc = subprocess.run(cmd,
                              capture_output=True, text=True, check=True)
        trajs.append(np.array(json.loads(proc.stdout.strip().splitlines()[-1])["trajectory"]))
    state_file.unlink()
    max_diff = float(np.abs(trajs[0] - trajs[1]).max())
    rep["cross_process"] = {"n_steps": 20, "max_abs_state_diff": max_diff}
    checks["cross_process_deterministic"] = rep["cross_process"]["max_abs_state_diff"] < 1e-9

    rep["checks"] = checks
    rep["all_passed"] = all(checks.values())
    return rep


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="lift", choices=["lift", "can"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out-dir", type=Path, default=feasibility_reports_dir())
    p.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.child:
        child_rollout(args.task, args.child)
        return 0
    rep = run(args.task, args.seed, args.steps, args.out_dir)
    out = args.out_dir / f"smoke_sim_{args.task}.json"
    out.write_text(json.dumps(rep, indent=1, default=str))
    for name, ok in rep["checks"].items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"report: {out}")
    return 0 if rep["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
