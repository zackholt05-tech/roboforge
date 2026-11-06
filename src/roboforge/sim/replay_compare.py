"""Demonstration replay through the research apparatus (Session 2; DECISIONS D-029).

Each sampled demonstration is restored from its own dataset XML and ``states[0]``. Its
recorded actions are played open-loop through :class:`RoboForgeEnv` and the rollout recorder,
with the policy labelled **non-autonomous**. The replay is compared with:

1. the recorded dataset states (the divergence summary: contact dynamics make open-loop
   replay drift, KI-004);
2. a bare robosuite loop (``reset_to_state`` + ``env.step``), which must be **bitwise**
   identical. This gate shows that the adapter, recorder and renderer add nothing to the
   physics;
3. Session 0's legacy ``replay_demo`` metrics (same platform: identical);
4. the controller bounds: dataset actions must produce zero clip events.

**Sampling rule (declared before the first run).** Sort all 200 demos by (length, index) and
cut them into 10 bins of 20. Draw one demo per bin with ``np.random.default_rng(0)``. Add the
lowest-index demo that failed Session 0's open-loop replay (Lift ``demo_177``, Can
``demo_11``). Then add the lowest-index validation-split demos until at least two are in the
sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from roboforge.eval.policies import DemoReplayPolicy
from roboforge.eval.rollout import RenderSpec, run_episode
from roboforge.sim.env_adapter import RoboForgeEnv
from roboforge.sim.snapshot import from_demo
from roboforge.sim.state_replay import reset_to_state, state_errors, state_layout

N_BINS = 10
SAMPLE_SEED = 0
SESSION0_FIRST_FAILURE = {"lift": "demo_177", "can": "demo_11"}
MIN_VALID = 2


def select_demos(lengths: dict[str, int], valid_ids: list[str], task: str) -> list[dict[str, Any]]:
    """Apply the declared sampling rule; return ``[{demo, reason}]`` in selection order."""
    order = sorted(lengths, key=lambda d: (lengths[d], int(d.split("_")[1])))
    bins = np.array_split(np.array(order), N_BINS)
    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = [{"demo": str(rng.choice(b)), "reason": f"length bin {i}"}
              for i, b in enumerate(bins)]
    names = {c["demo"] for c in chosen}
    fail = SESSION0_FIRST_FAILURE[task]
    if fail not in names:
        chosen.append({"demo": fail, "reason": "lowest-index Session 0 replay failure"})
        names.add(fail)
    for d in sorted(valid_ids, key=lambda x: int(x.split("_")[1])):
        if sum(c["demo"] in valid_ids for c in chosen) >= MIN_VALID:
            break
        if d not in names:
            chosen.append({"demo": d, "reason": "ensure >= 2 validation-split demos"})
            names.add(d)
    return chosen


def bare_replay(raw_env: Any, model_xml: str, states: np.ndarray,
                actions: np.ndarray) -> np.ndarray:
    """States after each action from a bare robosuite loop (the bitwise reference)."""
    out = [reset_to_state(raw_env, model_xml, states[0])]
    for a in actions:
        raw_env.step(a)
        out.append(np.array(raw_env.sim.get_state().flatten()))
    return np.asarray(out)


def replay_one(env: RoboForgeEnv, task: str, demo: str, grp: Any, partition: str,
               dataset_sha256: str, task_objects: list[str],
               render: RenderSpec | None = None) -> tuple[dict[str, Any], Any, Any]:
    """Replay one demo through the adapter; return (summary, record, initial state)."""
    model_file, states, actions = grp.attrs["model_file"], grp["states"][()], grp["actions"][()]
    init = from_demo(task, demo, model_file, states, partition, dataset_sha256)
    portable_ok = env.raw_env.edit_model_xml(init.model_xml) == env.raw_env.edit_model_xml(
        model_file)
    policy = DemoReplayPolicy(actions, demo, dataset_sha256)
    rec = run_episode(env, policy, init, run_id=f"demo_replay_{task}", render=render)
    replayed = rec.arrays["state"]
    bare = bare_replay(env.raw_env, model_file, states, actions)
    layout = state_layout(env.raw_env).restricted_to(task_objects)
    per_step = [state_errors(layout, replayed[k], states[k]) for k in range(1, len(states))]
    obj = np.array([p["object_pos_max_abs_m"] for p in per_step])
    robot = np.array([p["robot_qpos_max_abs_rad"] for p in per_step])
    eef_err = np.linalg.norm(rec.arrays["obs"][1:len(states), 0:3]
                             - _features(env, states)[1:, 0:3], axis=1)
    recorded_success = [env_success_on(env, s) for s in states]
    summary = {
        "demo": demo, "partition": partition, "n_actions": int(len(actions)),
        "portable_xml_compiles_identically": bool(portable_ok),
        "bitwise_equal_to_bare_robosuite_loop": bool(
            replayed.shape == bare.shape and np.array_equal(replayed, bare)),
        "clip_events": int(rec.arrays["clip_mask"].sum()),
        "replay_success_any": rec.outcome["success"],
        "replay_first_success_step": rec.outcome["first_success_step"],
        "replay_first_success_step_robosuite_raw":
            rec.outcome["first_success_step_robosuite_raw"],
        "replay_success_final_frame": bool(rec.arrays["success"][-1]),
        "recorded_first_success_index": next((k for k, s in enumerate(recorded_success) if s),
                                             None),
        "recorded_final_state_success": bool(recorded_success[-1]),
        "object_pos_err_m": {"max": float(obj.max()), "final": float(obj[-1])},
        "robot_qpos_err_rad": {"max": float(robot.max()), "final": float(robot[-1])},
        "eef_pos_err_m": {"max": float(eef_err.max()), "final": float(eef_err[-1])},
        "object_pos_err_curve_m": obj.tolist(),
        "policy": rec.meta["policy"], "stop_reason": rec.outcome["stop_reason"]}
    return summary, rec, init


def _features(env: RoboForgeEnv, states: np.ndarray) -> np.ndarray:
    from roboforge.data.features import FeatureExtractor

    return FeatureExtractor(env.observation_spec).bind(env.model).from_states(states)


def env_success_on(env: RoboForgeEnv, state: np.ndarray) -> bool:
    """Success predicate of a recorded state, on the currently loaded model (episode over)."""
    sim = env.raw_env.sim
    sim.set_state_from_flattened(state)
    env._mj.mj_kinematics(env.model, sim.data._data)
    return bool(env.raw_env._check_success())
