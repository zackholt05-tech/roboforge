"""Closed-loop rollout runner and frame-indexed recorder (MASTER_SPEC §4.5; DECISIONS D-028).

**Frame convention.** Frame ``k`` (``k = 0..T``) is the simulator state after ``k`` executed
actions; frame 0 is the restored initial snapshot. Action ``t`` (``t = 0..T−1``) is chosen from
the observation of frame ``t`` and produces frame ``t + 1``. Per-frame arrays have ``T + 1``
rows; per-action arrays have ``T`` rows, and ``reward[t]`` / ``success[t + 1]`` describe the
frame that action ``t`` produced.

**Observation hook (Session 3, D-032).** If the policy defines ``observe(obs)``, the runner
calls it with frame 0's observation after the reset and with every new frame's observation
right after each executed action, including actions inside a chunk. A chunked policy with
an observation history therefore sees every frame, not only the frames it decides at. The
hook only feeds the policy; it never touches the simulator. An exception in it ends the
episode as a recorded error.

**Per-episode policy provenance (Session 4).** If the policy defines ``episode_info()``, its
result is stored as ``policy_episode`` in the record metadata (e.g. the noise seed of a
stochastic policy). It never enters the arrays, so ``physics_sha256`` is unaffected.

Every frame stores the flattened MuJoCo state, so everything else (features, kinematics, success,
renders) can be re-derived and audited. :func:`verify_record` does exactly that: it
re-simulates the executed actions from the snapshot and re-evaluates success on the recorded
states.

Records are ``record.json`` (metadata, outcome, array index) plus ``arrays.npz`` (compressed).
``arrays_sha256`` hashes (name, dtype, shape, bytes) in sorted name order, so it does not
depend on zip metadata. ``physics_sha256`` is the same hash without wall-clock arrays: two
runs of the same episode on the same platform must give the same ``physics_sha256``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roboforge.provenance import evaluation_platform, git_revision, package_versions
from roboforge.sim.env_adapter import InvalidActionError, RoboForgeEnv
from roboforge.sim.snapshot import InitialState

RECORD_SCHEMA = "roboforge.rollout/1"
FRAME_CONVENTION = ("frame k = state after k executed actions (frame 0 = restored snapshot); "
                    "action t is chosen from the observation of frame t and produces frame t+1")


# Wall-clock fields: excluded from ``physics_sha256`` (they differ between identical runs).
TIMING_ARRAYS = ("planning_latency_ms",)


def arrays_sha256(arrays: dict[str, np.ndarray], exclude: tuple[str, ...] = ()) -> str:
    """Canonical content hash of a dict of arrays (optionally excluding some names)."""
    h = hashlib.sha256()
    for name in sorted(k for k in arrays if k not in exclude):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype.str}|{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class RenderSpec:
    """Which frames to render into the record (shadow renderer; never affects physics)."""

    cameras: tuple[str, ...]
    height: int = 128
    width: int = 128
    every: int = 1


@dataclass
class RolloutRecord:
    """One recorded episode: metadata + outcome + frame-indexed arrays."""

    meta: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(repr=False)

    @property
    def outcome(self) -> dict[str, Any]:
        """Outcome block (success, first success step, stop reason, ...)."""
        return self.meta["outcome"]

    def save(self, out_dir: Path) -> Path:
        """Write ``record.json`` + ``arrays.npz`` into ``out_dir``; return the JSON path."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "arrays.npz", **self.arrays)
        path = out_dir / "record.json"
        path.write_text(json.dumps(self.meta, indent=1, sort_keys=True, allow_nan=False) + "\n")
        return path

    @classmethod
    def load(cls, out_dir: Path) -> RolloutRecord:
        """Read a saved record and verify its array hash."""
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "record.json").read_text())
        with np.load(out_dir / "arrays.npz") as z:
            arrays = {k: z[k] for k in z.files}
        if arrays_sha256(arrays) != meta["arrays_sha256"]:
            raise ValueError(f"{out_dir}: arrays do not match arrays_sha256")
        return cls(meta, arrays)


def _outcome(success: np.ndarray, phase: np.ndarray, terminated: bool, truncated: bool,
             stop: str, error: dict[str, str] | None, clip: np.ndarray,
             raw_success: np.ndarray) -> dict[str, Any]:
    ep = np.flatnonzero(phase == 0) + 1  # frames produced by episode-phase actions
    hold = np.flatnonzero(phase == 1) + 1
    hits = [int(k) for k in ep if success[k]]
    raw_hits = [int(k) for k in ep if raw_success[k]]
    return {"success": bool(hits), "first_success_step": hits[0] if hits else None,
            "first_success_step_robosuite_raw": raw_hits[0] if raw_hits else None,
            "steps": int(len(ep)), "terminated": bool(terminated), "truncated": bool(truncated),
            "stop_reason": stop, "error": error,
            "hold_steps_executed": int(len(hold)),
            "success_held_through_hold": bool(len(hold) and success[hold].all()) or None,
            "n_actions_with_clip": int(clip.any(axis=1).sum()) if len(clip) else 0,
            "n_clipped_entries": int(clip.sum())}


def run_episode(env: RoboForgeEnv, policy: Any, initial_state: InitialState, *,
                run_id: str = "adhoc", episode_index: int = 0, codec: Any = None,
                render: RenderSpec | None = None) -> RolloutRecord:
    """Run ``policy`` closed-loop from ``initial_state`` and record every frame.

    Policies in ``"normalized"`` action space are decoded with ``codec``
    (:class:`roboforge.data.normalization.ActionCodec`). Errors (invalid actions, policy
    exceptions) stop the episode and are recorded as a failed episode with the error,
    never dropped.
    """
    obs, meta = env.reset(initial_state)
    policy.reset_episode(meta)
    observe = getattr(policy, "observe", None)
    kin = env.kinematics
    frames: dict[str, list[Any]] = {k: [] for k in (
        "state", "obs", "time_s", "joint_qpos", "joint_qvel", "body_pos", "body_quat_wxyz",
        "success", "success_robosuite_raw")}
    steps: dict[str, list[Any]] = {k: [] for k in (
        "raw_policy_action", "executed_action", "clip_mask", "reward", "reward_robosuite_raw",
        "decision_index", "chunk_offset", "planning_latency_ms", "phase")}
    rgb: dict[str, list[np.ndarray]] = {c: [] for c in (render.cameras if render else ())}
    rgb_index: list[int] = []

    def add_frame(o: Any, success: bool, raw_success: bool) -> None:
        state = env.current_state()
        f = kin.frame(state)
        for k, v in (("state", state), ("obs", o.vector), ("time_s", o.time_s),
                     ("success", success), ("success_robosuite_raw", raw_success),
                     *f.items()):
            frames[k].append(v)
        k_idx = len(frames["state"]) - 1
        if render and k_idx % render.every == 0:
            rgb_index.append(k_idx)
            for c in render.cameras:
                rgb[c].append(env.render(c, render.width, render.height))

    add_frame(obs, meta.initial_success, meta.initial_success)
    terminated = truncated = False
    stop, error, decision = "running", None, 0
    if observe is not None:
        try:
            observe(obs)
        except Exception as exc:  # recorded, never swallowed
            stop, error = "error", {"type": type(exc).__name__, "message": str(exc)}
    while stop == "running":
        t0 = time.perf_counter()
        try:
            chunk = np.atleast_2d(np.asarray(policy.act(obs), dtype=np.float64))
        except StopIteration:
            stop = "policy_exhausted"
            break
        except Exception as exc:  # recorded, never swallowed: the episode counts as failed
            stop, error = "error", {"type": type(exc).__name__, "message": str(exc)}
            break
        latency = 1000.0 * (time.perf_counter() - t0)
        for j, raw in enumerate(chunk):
            codec_clip = np.zeros(env.action_dim, dtype=bool)
            env_action = raw
            if policy.info.action_space == "normalized":
                env_action, codec_clip = codec.decode(raw)
            try:
                obs, reward, terminated, truncated, info = env.step(env_action)
            except InvalidActionError as exc:
                stop, error = "error", {"type": type(exc).__name__, "message": str(exc)}
                break
            for k, v in (("raw_policy_action", raw), ("executed_action", info["executed_action"]),
                         ("clip_mask", info["clip_mask"] | codec_clip), ("reward", reward),
                         ("reward_robosuite_raw", info["reward_robosuite_raw"]),
                         ("decision_index", decision), ("chunk_offset", j),
                         ("planning_latency_ms", latency if j == 0 else np.nan),
                         ("phase", 1 if info["phase"] == "hold" else 0)):
                steps[k].append(v)
            add_frame(obs, info["success"], info["success_robosuite_raw"])
            if observe is not None:
                try:
                    observe(obs)
                except Exception as exc:  # recorded, never swallowed
                    stop, error = "error", {"type": type(exc).__name__, "message": str(exc)}
                    break
            hold_done = terminated and info["phase"] == "hold" and env.hold_remaining == 0
            if truncated or (terminated and env.hold_steps == 0) or hold_done:
                stop = "terminated_success" if terminated else "truncated_horizon"
                break
        decision += 1

    a_dim = env.action_dim
    arrays = {k: np.asarray(v) for k, v in frames.items()}
    arrays.update({
        "raw_policy_action": np.asarray(steps["raw_policy_action"]).reshape(-1, a_dim),
        "executed_action": np.asarray(steps["executed_action"]).reshape(-1, a_dim),
        "clip_mask": np.asarray(steps["clip_mask"], dtype=bool).reshape(-1, a_dim),
        "reward": np.asarray(steps["reward"], dtype=np.float64),
        "reward_robosuite_raw": np.asarray(steps["reward_robosuite_raw"], dtype=np.float64),
        "decision_index": np.asarray(steps["decision_index"], dtype=np.int64),
        "chunk_offset": np.asarray(steps["chunk_offset"], dtype=np.int64),
        "planning_latency_ms": np.asarray(steps["planning_latency_ms"], dtype=np.float64),
        "phase": np.asarray(steps["phase"], dtype=np.uint8)})
    for c, imgs in rgb.items():
        arrays[f"rgb_{c}"] = np.asarray(imgs, dtype=np.uint8)
    if render:
        arrays["rgb_frame_index"] = np.asarray(rgb_index, dtype=np.int64)
    outcome = _outcome(arrays["success"], arrays["phase"], terminated, truncated, stop, error,
                       arrays["clip_mask"], arrays["success_robosuite_raw"])
    record_meta = {
        "schema": RECORD_SCHEMA, "run_id": run_id, "episode_index": int(episode_index),
        "task": env.task, "policy": policy.info.to_dict(), "episode": asdict(meta),
        "platform": evaluation_platform(), "git": git_revision(),
        "simulator": {"engine": "mujoco", "packages": package_versions(("mujoco", "robosuite")),
                      "env_name": env.sim_config["env_name"],
                      "control_mode": env.action_spec.controller,
                      "control_dt_s": env.control_dt, "physics_dt_s": meta.physics_dt_s,
                      "horizon": env.horizon, "termination": env.termination,
                      "hold_steps": env.hold_steps},
        "frame_convention": FRAME_CONVENTION,
        "quaternion_convention": "body_quat_wxyz: MuJoCo xquat (w, x, y, z), world frame",
        "bodies": {"names": kin.body_names,
                   "categories": kin.body_categories},
        "joints": kin.joint_names,
        "observation_names": [f.name for f in env.observation_spec.features],
        "render": asdict(render) if render else None,
        "arrays": {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in arrays.items()},
        "arrays_sha256": arrays_sha256(arrays),
        "physics_sha256": arrays_sha256(arrays, exclude=TIMING_ARRAYS), "outcome": outcome}
    episode_info = getattr(policy, "episode_info", None)
    if episode_info is not None:  # per-episode policy provenance, e.g. a sampler noise seed
        record_meta["policy_episode"] = episode_info()
    return RolloutRecord(record_meta, arrays)


def verify_record(record: RolloutRecord, initial_state: InitialState,
                  env: RoboForgeEnv | None = None) -> dict[str, Any]:
    """Re-simulate a record's executed actions and re-evaluate success on its stored states.

    Returns maximum absolute differences and pass flags. Same-platform expectation: all 0.0.
    """
    own = env is None
    env = env or RoboForgeEnv(record.meta["task"], horizon=record.meta["simulator"]["horizon"],
                              hold_steps=record.meta["simulator"]["hold_steps"])
    a = record.arrays
    try:
        obs, _ = env.reset(initial_state)
        states, observations, success = [env.current_state()], [obs.vector], []
        for act in a["executed_action"]:
            obs, _, _, _, info = env.step(act)
            states.append(env.current_state())
            observations.append(obs.vector)
            success.append(info["success"])
        resim = np.asarray(states)
        sim = env.raw_env.sim
        reeval = []
        for s in a["state"]:  # episode over: free to overwrite the live data now
            sim.set_state_from_flattened(s)
            env._mj.mj_kinematics(env.model, sim.data._data)
            reeval.append(bool(env.raw_env._check_success()))
        kin = np.asarray([env.kinematics.frame(s)["body_pos"] for s in a["state"]])
    finally:
        if own:
            env.close()
    out = {"resim_state_max_abs": float(np.abs(resim - a["state"]).max()),
           "resim_obs_max_abs": float(np.abs(np.asarray(observations) - a["obs"]).max()),
           "resim_success_equal": success == a["success"][1:].tolist(),
           "reevaluated_success_equal": reeval == a["success"].tolist(),
           "body_pos_max_abs": float(np.abs(kin - a["body_pos"]).max()),
           "outcome_consistent": record.outcome["success"] == bool(
               a["success"][1:][a["phase"] == 0].any())}
    out["passed"] = (out["resim_state_max_abs"] == 0.0 and out["resim_obs_max_abs"] == 0.0
                     and out["resim_success_equal"] and out["reevaluated_success_equal"]
                     and out["body_pos_max_abs"] == 0.0 and out["outcome_consistent"])
    return out
