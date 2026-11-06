"""``RoboForgeEnv``: the typed research adapter over robosuite v1.5.1 (MASTER_SPEC §4.1).

The adapter wraps a robosuite environment built from the dataset's own controller settings
and enforces the Session 1 and Session 2 contracts (DECISIONS D-010, D-024..D-026):

* **Resets** always go through an exact serialized snapshot (:mod:`roboforge.sim.snapshot`):
  an integer seed is first turned into a snapshot, then restored like a stored one.
  Seeds in the final-test block are refused until the freeze (:mod:`roboforge.sim.eval_states`).
* **Actions** must have exactly ``ActionSpec.dim`` finite entries (read from the committed task
  manifest, never hard-coded). They are clipped to the ActionSpec bounds, and the raw action,
  executed action and clip mask are reported.
* **Observations** are ``FeatureExtractor.from_sim`` of the state after the step, never
  robosuite's ``env.step`` observations (KI-015).
* **Success** is robosuite's ``_check_success`` evaluated on *current* kinematics:
  ``mj_kinematics`` is run on the live data first. ``mj_kinematics`` is inert for the dynamics;
  ``mj_forward`` is not, because it rewrites MuJoCo's solver warm start. The robosuite value
  on stale kinematics is reported alongside as ``success_robosuite_raw``.
* **Reward** is the sparse task reward of these configs (``reward_shaping=False``,
  ``reward_scale=1``): 1.0 when the success predicate holds, else 0.0, evaluated on the same
  current kinematics. robosuite's own step reward (stale kinematics) is reported as
  ``reward_robosuite_raw``.
* **Termination** (D-026): ``terminated`` at the first successful step; ``truncated`` when the
  horizon (task manifest, 400 steps) is reached without success. Further steps raise, except
  for an optional, explicitly requested post-success hold (persistence diagnostic only).
* **Kinematics and renders** come from scratch ``MjData`` (:mod:`roboforge.sim.kinematics`,
  :mod:`roboforge.sim.shadow_render`), so recording and rendering cannot change the rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from roboforge.data.features import FeatureExtractor
from roboforge.data.pipeline import load_specs, load_task_artifacts
from roboforge.sim.eval_states import check_seed_allowed, suites_config
from roboforge.sim.kinematics import KinematicsExtractor
from roboforge.sim.shadow_render import ShadowRenderer
from roboforge.sim.snapshot import InitialState, generate, restore


class InvalidActionError(ValueError):
    """An action with the wrong shape or non-finite entries (never silently repaired)."""


class EpisodeDoneError(RuntimeError):
    """``step`` after the episode terminated or was truncated."""


@dataclass(frozen=True)
class Observation:
    """Policy input at one frame: the canonical feature vector (float64, spec order)."""

    vector: np.ndarray
    step_index: int
    time_s: float


@dataclass(frozen=True)
class EpisodeMetadata:
    """What every rollout record states about its episode."""

    task: str
    initial_state: dict[str, Any]
    horizon: int
    hold_steps: int
    termination: str
    control_dt_s: float
    physics_dt_s: float
    observation_spec_sha256: str
    action_spec_sha256: str
    task_spec_sha256: str
    initial_success: bool


class RoboForgeEnv:
    """Research adapter: exact resets, validated actions, canonical observations, true success."""

    def __init__(self, task: str, *, horizon: int | None = None, hold_steps: int = 0) -> None:
        import mujoco

        from roboforge.sim.factory import load_sim_config, make_env

        self._mj = mujoco
        self.task = task
        self.sim_config = load_sim_config(task)
        if self.sim_config["env_kwargs"].get("reward_shaping"):
            raise NotImplementedError("shaped rewards are not part of the RoboForge contract")
        manifest = load_task_artifacts(task)["task_manifest"]
        self.task_spec = manifest["task_spec"]
        self._hashes = {k: manifest[k] for k in ("observation_spec_sha256",
                                                 "action_spec_sha256", "task_spec_sha256")}
        self.observation_spec, self.action_spec = load_specs(task)
        self.low = np.array(self.action_spec.low, dtype=np.float64)
        self.high = np.array(self.action_spec.high, dtype=np.float64)
        self.horizon = int(horizon or self.task_spec["eval_horizon_steps"])
        self.hold_steps = int(hold_steps)
        self.termination = suites_config()["episode"]["termination"]
        self._env = make_env(self.sim_config)
        lo, hi = self._env.action_spec
        if (self._env.action_dim != self.action_spec.dim or not np.array_equal(lo, self.low)
                or not np.array_equal(hi, self.high)):
            raise AssertionError("simulator action space differs from the committed ActionSpec")
        self._features = FeatureExtractor(self.observation_spec)
        self._kin: KinematicsExtractor | None = None
        self._shadow: ShadowRenderer | None = None
        self._init: InitialState | None = None
        self._t = 0
        self._terminated = self._truncated = False
        self._hold_left = 0

    # ------------------------------------------------------------------ properties
    @property
    def action_dim(self) -> int:
        """Action dimension from the committed ActionSpec."""
        return self.action_spec.dim

    @property
    def control_dt(self) -> float:
        """Seconds per control step."""
        return float(self._env.control_timestep)

    @property
    def raw_env(self) -> Any:
        """The underlying robosuite env (read-only use in checks; do not step it directly)."""
        return self._env

    @property
    def hold_remaining(self) -> int:
        """Post-success hold steps still allowed in this episode."""
        return self._hold_left

    @property
    def model(self) -> Any:
        """The compiled ``mujoco.MjModel`` of the current episode."""
        return self._env.sim.model._model

    # ------------------------------------------------------------------ episode control
    def _resolve(self, initial_state_or_seed: InitialState | int) -> InitialState:
        if isinstance(initial_state_or_seed, InitialState):
            if initial_state_or_seed.task != self.task:
                raise ValueError(f"state for {initial_state_or_seed.task!r}, env is {self.task!r}")
            return initial_state_or_seed
        seed = int(initial_state_or_seed)
        partition = check_seed_allowed(self.task, seed)
        return generate(self._env, self.task, seed, f"{self.task}_seed{seed}", partition)

    def reset(self, initial_state_or_seed: InitialState | int) -> tuple[Observation,
                                                                        EpisodeMetadata]:
        """Restore an exact snapshot (or a seed's snapshot); return t = 0 observation + metadata."""
        init = self._resolve(initial_state_or_seed)
        restore(self._env, init)
        self._init, self._t = init, 0
        self._terminated = self._truncated = False
        self._hold_left = self.hold_steps
        if self._kin is None or self._kin.model is not self.model:
            self._kin = KinematicsExtractor(self.model)
            if self._shadow is not None:
                self._shadow.close()
                self._shadow = None
        meta = EpisodeMetadata(
            task=self.task, initial_state=init.reference(), horizon=self.horizon,
            hold_steps=self.hold_steps, termination=self.termination,
            control_dt_s=self.control_dt, physics_dt_s=float(self._env.model_timestep),
            initial_success=self.get_success(), **self._hashes)
        return self._observation(), meta

    def _observation(self) -> Observation:
        return Observation(vector=self._features.from_sim(self._env.sim), step_index=self._t,
                           time_s=float(self._env.sim.data.time))

    def validate_action(self, action: Any) -> np.ndarray:
        """Return ``action`` as float64 ``[dim]``; raise :class:`InvalidActionError` otherwise."""
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (self.action_dim,):
            raise InvalidActionError(f"action shape {a.shape} != ({self.action_dim},)")
        if not np.all(np.isfinite(a)):
            raise InvalidActionError(f"non-finite action {a.tolist()}")
        return a

    def step(self, action: Any) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply one clipped action for one control step (25 physics substeps)."""
        if self._init is None:
            raise RuntimeError("call reset() before step()")
        in_hold = False
        if self._terminated or self._truncated:
            if not (self._terminated and self._hold_left > 0):
                raise EpisodeDoneError("episode is over (terminated or truncated)")
            in_hold, self._hold_left = True, self._hold_left - 1
        raw = self.validate_action(action)
        executed = np.clip(raw, self.low, self.high)
        _, reward_raw, _, _ = self._env.step(executed)
        success_raw = bool(self._env._check_success())  # robosuite: pre-substep kinematics
        success = self.get_success()
        self._t += 1
        if not in_hold:
            self._terminated = success
            self._truncated = self._t >= self.horizon and not success
        info = {"step_index": self._t, "phase": "hold" if in_hold else "episode",
                "success": success, "success_robosuite_raw": success_raw,
                "reward_robosuite_raw": float(reward_raw), "raw_action": raw,
                "executed_action": executed, "clip_mask": executed != raw}
        return (self._observation(), 1.0 if success else 0.0, self._terminated,
                self._truncated, info)

    def get_success(self) -> bool:
        """Task success predicate on current kinematics (``mj_kinematics`` only; inert)."""
        self._mj.mj_kinematics(self.model, self._env.sim.data._data)
        return bool(self._env._check_success())

    # ------------------------------------------------------------------ state and kinematics
    def current_state(self) -> np.ndarray:
        """Flattened MuJoCo state ``[time, qpos, qvel]``."""
        return np.array(self._env.sim.get_state().flatten())

    def serialize_initial_state(self) -> dict[str, Any]:
        """The current episode's snapshot (portable XML + state + labels)."""
        if self._init is None:
            raise RuntimeError("no episode has been reset")
        return self._init.to_dict()

    @property
    def kinematics(self) -> KinematicsExtractor:
        """Extractor bound to the current model."""
        if self._kin is None:
            raise RuntimeError("call reset() first")
        return self._kin

    def get_robot_kinematics(self) -> dict[str, Any]:
        """Joint names/angles/velocities and robot link world transforms (wxyz quaternions)."""
        return self.kinematics.robot(self.current_state())

    def get_scene_kinematics(self) -> dict[str, Any]:
        """World transforms of objects and fixtures (wxyz quaternions)."""
        return self.kinematics.scene(self.current_state())

    def render(self, camera_id: str, width: int, height: int) -> np.ndarray:
        """Upright RGB frame of the current state (scratch data; does not touch the rollout)."""
        if self._shadow is None:
            self._shadow = ShadowRenderer(self.model)
        return self._shadow.rgb(self.current_state(), camera_id, height, width)

    def constants(self) -> dict[str, Any]:
        """Semantic constants read from the loaded model and committed specs (§4.1)."""
        robot = self._env.robots[0]
        kin = self.kinematics
        m = self.model
        return {
            "task": self.task, "env_name": self.sim_config["env_name"], "robot": robot.name,
            "arm_joints": list(robot.robot_joints),
            "gripper_joints": list(robot.gripper_joints["right"]),
            "actuators": [self._mj.mj_id2name(m, self._mj.mjtObj.mjOBJ_ACTUATOR, i)
                          for i in range(m.nu)],
            "control_freq_hz": float(self._env.control_freq), "control_dt_s": self.control_dt,
            "physics_dt_s": float(self._env.model_timestep),
            "substeps_per_control": round(self.control_dt / float(self._env.model_timestep)),
            "action_spec": self.action_spec.to_dict(),
            "observation_slices": {k: list(v) for k, v in self.observation_spec.slices().items()},
            "privileged_features": [f.name for f in self.observation_spec.features
                                    if f.privileged],
            "success_predicate": self.task_spec["success_predicate"],
            "horizon": self.horizon, "termination": self.termination,
            "cameras": {c: float(m.cam_fovy[i]) for i, c in enumerate(kin.camera_names)},
            "link_hierarchy": [b.__dict__ for b in kin.hierarchy()],
            "model_dims": {"nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
                           "nbody": int(m.nbody), "nmocap": int(m.nmocap)},
        }

    def close(self) -> None:
        """Release the simulator and GL contexts."""
        if self._shadow is not None:
            self._shadow.close()
            self._shadow = None
        self._env.close()
