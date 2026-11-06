"""Exact state restoration and open-loop demonstration replay.

Two distinct checks (both required by MASTER_SPEC §2.3 item 4):

1. **State restoration** – load the episode's saved MJCF (``model_file``) and set
   the flattened MuJoCo state ``states[0]``; the simulator state read back must
   equal it (pure serialization round-trip, says nothing about the controller).
2. **Open-loop action replay** – from ``states[0]``, step the recorded dataset
   actions through the dataset's controller and compare every resulting state
   with ``states[t+1]``. This validates the controller/action mapping. Contact
   dynamics make some drift normal; errors are decomposed by component so a
   chaotic angular-velocity spike is not confused with a wrong controller.

Flattened state layout in robosuite v1.5.1 / MuJoCo 3.x: ``[time, qpos(nq), qvel(nv)]``.
Free-joint qpos is ``[x, y, z, qw, qx, qy, qz]`` (MuJoCo scalar-first quaternions).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

MJ_JNT_FREE = 0  # mujoco.mjtJoint.mjJNT_FREE


@dataclass(frozen=True)
class StateLayout:
    """Index map of the flattened MuJoCo state vector."""

    nq: int
    nv: int
    robot_qpos_idx: list[int]
    robot_qvel_idx: list[int]
    object_pos_idx: list[list[int]]
    object_quat_idx: list[list[int]]
    object_names: list[str]

    @property
    def state_dim(self) -> int:
        """Length of ``[time, qpos, qvel]``."""
        return 1 + self.nq + self.nv

    def restricted_to(self, object_names: list[str]) -> StateLayout:
        """Return a copy whose object metrics cover only ``object_names`` (free-joint names)."""
        missing = set(object_names) - set(self.object_names)
        if missing:
            raise KeyError(f"free joints {sorted(missing)} not in model {self.object_names}")
        keep = [i for i, n in enumerate(self.object_names) if n in object_names]
        return StateLayout(self.nq, self.nv, self.robot_qpos_idx, self.robot_qvel_idx,
                           [self.object_pos_idx[i] for i in keep],
                           [self.object_quat_idx[i] for i in keep],
                           [self.object_names[i] for i in keep])


def state_layout(env: Any) -> StateLayout:
    """Derive the flattened-state index layout from the loaded MuJoCo model."""
    m = env.sim.model._model
    robot_q, robot_v, obj_pos, obj_quat, names = [], [], [], [], []
    for j in range(m.njnt):
        qadr, vadr = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        if int(m.jnt_type[j]) == MJ_JNT_FREE:
            obj_pos.append([1 + qadr + k for k in range(3)])
            obj_quat.append([1 + qadr + k for k in range(3, 7)])
            names.append(env.sim.model.joint_id2name(j))
        else:
            robot_q.append(1 + qadr)
            robot_v.append(1 + m.nq + vadr)
    return StateLayout(int(m.nq), int(m.nv), robot_q, robot_v, obj_pos, obj_quat, names)


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Rotation angle (rad) between unit quaternions, invariant to the q/-q antipode."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(q1, q2))), 0.0, 1.0)))


def state_errors(layout: StateLayout, actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """Decompose ``|actual - expected|`` into physically meaningful components."""
    diff = np.abs(actual - expected)
    qvel_slice = slice(1 + layout.nq, 1 + layout.nq + layout.nv)
    return {
        "time_abs": float(diff[0]),
        "robot_qpos_max_abs_rad": float(diff[layout.robot_qpos_idx].max()),
        "object_pos_max_abs_m": max(
            (float(diff[i].max()) for i in layout.object_pos_idx), default=0.0
        ),
        "object_rot_max_rad": max(
            (quat_angle(actual[i], expected[i]) for i in layout.object_quat_idx), default=0.0
        ),
        "qvel_max_abs": float(diff[qvel_slice].max()),
        "robot_qvel_max_abs": float(diff[layout.robot_qvel_idx].max()),
        "full_state_max_abs": float(diff.max()),
    }


def reset_to_state(env: Any, model_xml: str, state: np.ndarray) -> np.ndarray:
    """Hard-reset ``env`` to the saved model XML and flattened state; return the read-back state.

    Mirrors robomimic ``EnvRobosuite.reset_to`` for robosuite >= 1.4.
    """
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(model_xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    return np.array(env.sim.get_state().flatten())


def is_success(env: Any) -> bool:
    """Return the task's own success predicate (robosuite ``_check_success``)."""
    return bool(env._check_success())


@dataclass
class ReplayResult:
    """Outcome of replaying one demonstration's actions open-loop."""

    demo: str
    n_actions: int
    reset_state_max_abs: float
    dataset_final_state_success: bool
    replay_success_at_end: bool
    replay_first_success_step: int | None
    max_errors: dict[str, float]
    final_errors: dict[str, float]
    control_steps_per_sec: float
    per_step: list[dict[str, float]] = field(default_factory=list, repr=False)

    def to_dict(self, include_per_step: bool = False) -> dict[str, Any]:
        """JSON-serializable form (per-step errors optional to keep reports small)."""
        d = asdict(self)
        if not include_per_step:
            d.pop("per_step")
        return d


def replay_demo(
    env: Any, demo: str, model_xml: str, states: np.ndarray, actions: np.ndarray,
    task_objects: list[str] | None = None,
) -> ReplayResult:
    """Replay ``actions`` from ``states[0]`` and compare with recorded ``states[1:]``.

    ``task_objects`` restricts object-pose metrics to those free joints (default: all).
    ``qvel_max_abs`` always covers the full velocity vector.
    """
    if len(states) != len(actions):
        raise ValueError(f"{demo}: {len(states)} states vs {len(actions)} actions")
    layout = state_layout(env)
    if task_objects is not None:
        layout = layout.restricted_to(task_objects)
    if layout.state_dim != states.shape[1]:
        raise ValueError(f"{demo}: model state dim {layout.state_dim} != dataset {states.shape[1]}")

    # Ground-truth label: does the recorded final state satisfy the task predicate?
    reset_to_state(env, model_xml, states[-1])
    final_state_success = is_success(env)

    achieved = reset_to_state(env, model_xml, states[0])
    reset_err = float(np.abs(achieved - states[0]).max())

    per_step: list[dict[str, float]] = []
    first_success: int | None = None
    t0 = time.perf_counter()
    for t, action in enumerate(actions):
        env.step(action)
        if first_success is None and is_success(env):
            first_success = t
        if t + 1 < len(states):
            per_step.append(state_errors(layout, env.sim.get_state().flatten(), states[t + 1]))
    elapsed = time.perf_counter() - t0
    keys = per_step[0].keys()
    return ReplayResult(
        demo=demo,
        n_actions=len(actions),
        reset_state_max_abs=reset_err,
        dataset_final_state_success=final_state_success,
        replay_success_at_end=is_success(env),
        replay_first_success_step=first_success,
        max_errors={k: max(p[k] for p in per_step) for k in keys},
        final_errors=per_step[-1],
        control_steps_per_sec=len(actions) / elapsed,
        per_step=per_step,
    )
