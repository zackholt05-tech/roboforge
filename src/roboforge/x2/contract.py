"""X2 (Can MH): the executable checks that the validated Can contract applies (D-121).

Option A of ``docs/extensions/X2_harder_task.md`` reuses the Can simulator contract validated
in Sessions 0–2 instead of building a new one. Before anything relies on it, this module shows
it applies, from the hash-verified MH files and the committed artifacts:

* **env_args.** The task-defining kwargs of both MH files equal ``configs/simulator/can.yaml``
  (the frozen ``task_kwargs``), with the same environment name and type.
* **Specs.** The X2 observation and action spec hashes equal the committed Can PH ones.
* **Task spec.** The X2 task spec differs from the Can PH one only in dataset-identity fields
  and the D-120 horizon (:data:`ALLOWED_TASK_SPEC_DIFFS`).
* **Replay.** Every MH demo is replayed open loop from its first saved state with the frozen
  :func:`roboforge.sim.state_replay.replay_demo` (restore error, replay success at the end,
  state errors), as Session 0 did for PH. Open-loop replay is an engineering check: it is
  chaotic in contact-rich tasks (KI-004), so its rate is reported, not gated.
* **Environment under the X2 binding.** Inside :func:`roboforge.x2.binding.x2_bindings`,
  ``RoboForgeEnv('can')`` has the X2 horizon and the committed action space.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

ALLOWED_TASK_SPEC_DIFFS = ("eval_horizon_steps", "n_demos", "total_samples", "source_files")


def task_spec_diff(x2_spec: dict[str, Any], ph_spec: dict[str, Any]) -> dict[str, Any]:
    """Fields where the X2 task spec differs from Can PH's, and whether all are allowed."""
    keys = sorted(set(x2_spec) | set(ph_spec))
    diff = {k: {"x2": x2_spec.get(k), "can_ph": ph_spec.get(k)} for k in keys
            if x2_spec.get(k) != ph_spec.get(k)}
    return {"differing_fields": diff,
            "only_allowed_fields_differ": set(diff) <= set(ALLOWED_TASK_SPEC_DIFFS)}


def env_args_check(paths: dict[str, Any]) -> dict[str, Any]:
    """Task kwargs, environment name and type of each MH file vs the Can simulator config."""
    from roboforge.data.inspect import env_args
    from roboforge.sim.factory import load_sim_config, task_kwargs

    sim = load_sim_config("can")
    out = {}
    for k, p in sorted(paths.items()):
        with h5py.File(p, "r") as f:
            ea = env_args(f)
        out[k] = {"task_kwargs_equal": task_kwargs(ea) == sim["env_kwargs"],
                  "env_name_equal": ea["env_name"] == sim["env_name"],
                  "type_equal": ea["type"] == sim["type"], "env_version": ea["env_version"]}
    return out


def _q(v: list[float]) -> dict[str, float]:
    a = np.asarray(v, dtype=float)
    return {"median": float(np.median(a)), "p95": float(np.quantile(a, 0.95)),
            "max": float(a.max())}


def replay_all(low_path: Any, operators: dict[str, str]) -> dict[str, Any]:
    """Open-loop replay of every demo (frozen ``replay_demo``), overall and per operator."""
    from roboforge.data.inspect import demo_keys
    from roboforge.sim.factory import load_sim_config, make_env
    from roboforge.sim.state_replay import replay_demo

    sim = load_sim_config("can")
    env = make_env(sim)
    rows = []
    try:
        with h5py.File(low_path, "r") as low:
            for d in demo_keys(low):
                g = low["data"][d]
                r = replay_demo(env, d, g.attrs["model_file"], g["states"][()], g["actions"][()],
                                task_objects=sim["task_object_joints"])
                rows.append({"demo": d, "operator": operators[d], "n_actions": r.n_actions,
                             "reset_state_max_abs": r.reset_state_max_abs,
                             "dataset_final_state_success": r.dataset_final_state_success,
                             "replay_success_at_end": r.replay_success_at_end,
                             "replay_first_success_step": r.replay_first_success_step,
                             "max_errors": r.max_errors, "final_errors": r.final_errors,
                             "control_steps_per_sec": r.control_steps_per_sec})
    finally:
        env.close()

    def agg(sel: list[dict[str, Any]]) -> dict[str, Any]:
        return {"n": len(sel), "replay_success": sum(r["replay_success_at_end"] for r in sel),
                "dataset_final_state_success": sum(r["dataset_final_state_success"]
                                                   for r in sel)}

    by_op = {op: agg([r for r in rows if r["operator"] == op])
             for op in sorted(set(operators.values()))}
    return {
        **agg(rows), "max_reset_state_error": max(r["reset_state_max_abs"] for r in rows),
        "failed_demos": [r["demo"] for r in rows if not r["replay_success_at_end"]],
        "per_operator": by_op,
        "max_object_pos_err_m": _q([r["max_errors"]["object_pos_max_abs_m"] for r in rows]),
        "final_object_pos_err_m": _q([r["final_errors"]["object_pos_max_abs_m"] for r in rows]),
        "max_robot_qpos_err_rad": _q([r["max_errors"]["robot_qpos_max_abs_rad"] for r in rows]),
        "control_steps_per_sec_median": float(np.median([r["control_steps_per_sec"]
                                                         for r in rows])),
        "per_demo": rows,
    }


def bound_env_check() -> dict[str, Any]:
    """``RoboForgeEnv('can')`` outside and inside the X2 binding: horizon, hashes, actions."""
    from roboforge.sim.env_adapter import RoboForgeEnv
    from roboforge.x2.binding import x2_bindings

    def probe() -> dict[str, Any]:
        env = RoboForgeEnv("can")
        try:
            return {"horizon": env.horizon, "hashes": dict(env._hashes),
                    "action_low": list(env.low), "action_high": list(env.high),
                    "observation_dim": env.observation_spec.dim}
        finally:
            env.close()

    outside = probe()
    with x2_bindings():
        inside = probe()
    return {"outside": outside, "inside": inside,
            "same_obs_action_specs": (
                outside["hashes"]["observation_spec_sha256"]
                == inside["hashes"]["observation_spec_sha256"]
                and outside["hashes"]["action_spec_sha256"]
                == inside["hashes"]["action_spec_sha256"]
                and outside["action_low"] == inside["action_low"]
                and outside["action_high"] == inside["action_high"])}
