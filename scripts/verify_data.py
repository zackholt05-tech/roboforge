"""Verify downloaded robomimic datasets against the pinned simulator (Session 0 evidence).

For each task this script:

1. re-verifies SHA-256 of the ``low_dim`` and ``raw`` files against the manifest;
2. summarizes HDF5 structure (keys, shapes, lengths, action ranges, upstream masks);
3. checks that the ``low_dim`` file's ``states``/``actions``/``model_file`` equal the raw file's;
4. checks observation/state alignment: restoring ``states[t]`` reproduces ``obs[t]``;
5. replays recorded actions open-loop for N demos and decomposes state error;
6. optionally renders frames of a replayed demonstration from named cameras.

Writes ``reports/feasibility/data_report_<task>.json`` (+ PNG frames with ``--render``).

Usage::

    python scripts/verify_data.py --tasks lift can --replay-demos all --render
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

BACKEND = configure_mujoco_gl()

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from roboforge.data.fetch import load_manifest, verify_file  # noqa: E402
from roboforge.data.inspect import demo_keys, env_args, summarize_hdf5  # noqa: E402
from roboforge.logging_utils import get_logger  # noqa: E402
from roboforge.paths import data_root, feasibility_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import environment_snapshot  # noqa: E402
from roboforge.sim.factory import load_sim_config, make_env  # noqa: E402
from roboforge.sim.renderer import camera_names, render_rgb  # noqa: E402
from roboforge.sim.state_replay import replay_demo, reset_to_state  # noqa: E402

LOG = get_logger("verify_data")
# Low-dim observation keys whose value is a deterministic function of the MuJoCo state.
ALIGNMENT_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos", "robot0_gripper_qpos",
                  "object")


def compare_lowdim_to_raw(low: h5py.File, raw: h5py.File) -> dict[str, Any]:
    """Check that the processed low_dim file carries the raw file's states/actions/models."""
    lk, rk = demo_keys(low), demo_keys(raw)
    out: dict[str, Any] = {"same_demo_keys": lk == rk}
    max_state, max_action, model_equal = 0.0, 0.0, True
    for d in lk:
        a, b = low["data"][d], raw["data"][d]
        max_state = max(max_state, float(np.abs(a["states"][()] - b["states"][()]).max()))
        max_action = max(max_action, float(np.abs(a["actions"][()] - b["actions"][()]).max()))
        model_equal &= a.attrs["model_file"] == b.attrs["model_file"]
    out.update(max_abs_state_diff=max_state, max_abs_action_diff=max_action,
               model_files_identical=bool(model_equal),
               masks_identical={k: sorted(low["mask"][k][()]) == sorted(raw["mask"][k][()])
                                for k in low["mask"]})
    return out


def check_obs_alignment(env: Any, grp: h5py.Group, timesteps: list[int],
                        object_blocks: dict[str, list[int]]) -> dict[str, float]:
    """Max |obs recomputed from states[t] - stored obs[t]| per key / object sub-block."""
    model_xml = grp.attrs["model_file"]
    states = grp["states"][()]
    keys = [k for k in ALIGNMENT_KEYS if k in grp["obs"] and k != "object"]
    errs = {k: 0.0 for k in keys} | {f"object.{b}": 0.0 for b in object_blocks}
    for t in timesteps:
        reset_to_state(env, model_xml, states[t])
        obs = env._get_observations(force_update=True)
        for k in keys:
            errs[k] = max(errs[k], float(np.abs(obs[k] - grp["obs"][k][t]).max()))
        # robomimic stores robosuite's "object-state" observable under the key "object".
        diff = np.abs(obs["object-state"] - grp["obs"]["object"][t])
        for b, (lo, hi) in object_blocks.items():
            errs[f"object.{b}"] = max(errs[f"object.{b}"], float(diff[lo:hi].max()))
    return errs


def render_demo_frames(env_args_: dict[str, Any], grp: h5py.Group, demo: str, task: str,
                       out_dir: Path, size: int = 256) -> dict[str, Any]:
    """Replay ``demo`` with an offscreen renderer and save frames at start/middle/end."""
    import imageio.v3 as iio

    env = make_env(env_args_, offscreen=True)
    cams = camera_names(env)
    use = [c for c in ("agentview", "frontview", "robot0_eye_in_hand") if c in cams]
    states, actions = grp["states"][()], grp["actions"][()]
    reset_to_state(env, grp.attrs["model_file"], states[0])
    picks = sorted({0, len(actions) // 2, len(actions) - 1})
    saved, render_ms = [], []
    for t in range(len(actions)):
        if t in picks:
            for cam in use:
                t0 = time.perf_counter()
                frame = render_rgb(env, cam, size, size)
                render_ms.append(1000 * (time.perf_counter() - t0))
                path = out_dir / f"{task}_{demo}_t{t:03d}_{cam}.png"
                iio.imwrite(path, frame)
                saved.append({"path": repo_relative(path), "t": t,
                              "camera": cam, "mean_pixel": float(frame.mean()),
                              "std_pixel": float(frame.std())})
        env.step(actions[t])
    success = bool(env._check_success())
    env.close()
    return {"backend": BACKEND, "cameras_in_model": cams, "frames": saved,
            "render_ms_mean": float(np.mean(render_ms)), "render_resolution": size,
            "success_after_replay_with_renderer": success}


def verify_task(task: str, replay_n: int | None, render: bool) -> dict[str, Any]:
    """Run every check for one task and return the report dictionary."""
    source, files = load_manifest()
    entries = {f.hdf5_type: f for f in files if f.task == task}
    report: dict[str, Any] = {"task": task, "environment": environment_snapshot(),
                              "manifest_source": source, "files": {}}
    for kind, entry in entries.items():
        path = entry.destination()
        if not path.exists():
            report["files"][kind] = {"present": False, "path": str(path)}
            continue
        report["files"][kind] = {"present": True, "path": str(path.relative_to(data_root())),
                                 "sha256": verify_file(entry, path),
                                 "sha256_matches_manifest": True,
                                 "summary": summarize_hdf5(path)}
    low_path, raw_path = entries["low_dim"].destination(), entries["raw"].destination()
    if not low_path.exists():
        raise FileNotFoundError(f"{low_path} missing; run `make download` first")

    with h5py.File(low_path, "r") as low:
        meta = env_args(low)
        if raw_path.exists():
            with h5py.File(raw_path, "r") as raw:
                report["lowdim_vs_raw"] = compare_lowdim_to_raw(low, raw)
        demos = demo_keys(low)
        env = make_env(meta)
        report["env_construction"] = {
            "env_name": meta["env_name"], "action_dim": int(env.action_dim),
            "action_low": env.action_spec[0].tolist(), "action_high": env.action_spec[1].tolist(),
            "control_freq_hz": env.control_freq, "control_timestep_s": env.control_timestep,
            "model_timestep_s": env.model_timestep, "horizon_default": env.horizon,
            "robot": env.robots[0].name, "robot_joints": list(env.robots[0].robot_joints),
            "nq": int(env.sim.model.nq), "nv": int(env.sim.model.nv), "nu": int(env.sim.model.nu),
        }
        g0 = low["data"][demos[0]]
        n = len(g0["actions"])
        sim_cfg = load_sim_config(task)
        report["obs_state_alignment_demo0"] = check_obs_alignment(
            env, g0, [0, n // 2, n - 1], sim_cfg["object_obs_blocks"])

        selected = demos if replay_n is None else demos[:replay_n]
        task_objects = sim_cfg["task_object_joints"]
        report["replay_task_object_joints"] = task_objects
        results = []
        for d in selected:
            g = low["data"][d]
            r = replay_demo(env, d, g.attrs["model_file"], g["states"][()], g["actions"][()],
                            task_objects=task_objects)
            results.append(r.to_dict())
            LOG.info("replayed", extra={"fields": {"task": task, "demo": d,
                     "success": r.replay_success_at_end,
                     "obj_pos_err": r.max_errors["object_pos_max_abs_m"]}})
        env.close()
        report["replay"] = aggregate_replay(results)
        report["replay"]["per_demo"] = results
        if render:
            out_dir = feasibility_reports_dir() / "frames"
            out_dir.mkdir(parents=True, exist_ok=True)
            report["render"] = render_demo_frames(meta, g0, demos[0], task, out_dir)
    report["integrity_checks"] = integrity_checks(report, sim_cfg)
    return report


def integrity_checks(report: dict[str, Any], sim_cfg: dict[str, Any]) -> dict[str, bool]:
    """Pass/fail checks that gate the exit code.

    Open-loop replay is chaotic in contact-rich tasks, so it is gated by the declared
    ``replay_success_rate_min`` guard instead of requiring every demo to replay.
    Lagged observation blocks (KNOWN_ISSUES KI-003) are excluded from the exact-alignment check.
    """
    r, lv = report["replay"], report.get("lowdim_vs_raw", {})
    lagged = {f"object.{b}" for b in sim_cfg["lagged_object_obs_blocks"]}
    aligned = {k: v for k, v in report["obs_state_alignment_demo0"].items() if k not in lagged}
    return {
        "hashes_match_manifest": all(f.get("sha256_matches_manifest", False)
                                     for f in report["files"].values() if f["present"]),
        "lowdim_equals_raw": bool(lv) and lv["same_demo_keys"] and lv["max_abs_state_diff"] == 0.0
        and lv["max_abs_action_diff"] == 0.0 and lv["model_files_identical"]
        and all(lv["masks_identical"].values()),
        "state_restore_exact": r["max_reset_state_error"] == 0.0,
        "obs_recomputed_from_state_exact": all(v <= 1e-9 for v in aligned.values()),
        "all_recorded_final_states_successful":
            r["dataset_final_state_success_count"] == r["n_replayed"],
        "replay_success_rate_meets_guard": r["replay_success_count"] / r["n_replayed"]
        >= sim_cfg["replay_success_rate_min"],
    }


def aggregate_replay(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-demo replay results into rates and error quantiles."""
    def q(key: str, which: str = "max_errors") -> dict[str, float]:
        v = np.array([r[which][key] for r in results])
        return {"median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)),
                "max": float(v.max())}

    return {
        "n_replayed": len(results),
        "replay_success_count": sum(r["replay_success_at_end"] for r in results),
        "dataset_final_state_success_count": sum(r["dataset_final_state_success"] for r in results),
        "max_reset_state_error": max(r["reset_state_max_abs"] for r in results),
        "failed_demos": [r["demo"] for r in results if not r["replay_success_at_end"]],
        "max_object_pos_err_m": q("object_pos_max_abs_m"),
        "max_robot_qpos_err_rad": q("robot_qpos_max_abs_rad"),
        "max_object_rot_err_rad": q("object_rot_max_rad"),
        "final_object_pos_err_m": q("object_pos_max_abs_m", "final_errors"),
        "max_qvel_err": q("qvel_max_abs"),
        "max_robot_qvel_err": q("robot_qvel_max_abs"),
        "control_steps_per_sec_median":
            float(np.median([r["control_steps_per_sec"] for r in results])),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=["lift"])
    p.add_argument("--replay-demos", default="10", help="number of demos to replay, or 'all'")
    p.add_argument("--render", action="store_true")
    p.add_argument("--out-dir", type=Path, default=feasibility_reports_dir())
    args = p.parse_args(argv)
    replay_n = None if args.replay_demos == "all" else int(args.replay_demos)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    for task in args.tasks:
        report = verify_task(task, replay_n, args.render)
        out = args.out_dir / f"data_report_{task}.json"
        out.write_text(json.dumps(report, indent=1, default=str))
        r = report["replay"]
        print(f"[{task}] replay success {r['replay_success_count']}/{r['n_replayed']} | "
              f"dataset final-state success "
              f"{r['dataset_final_state_success_count']}/{r['n_replayed']} | "
              f"obj pos err p95 {r['max_object_pos_err_m']['p95']:.4f} m | report {out}")
        for name, ok in report["integrity_checks"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not all(report["integrity_checks"].values()):
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
