"""The one RoboForge state -> observation-feature function (DECISIONS D-010, KI-003).

Every low-dimensional policy input is computed here, from the flattened MuJoCo state
``[time, qpos(nq), qvel(nv)]`` and the loaded MjModel, for both:

* dataset windows (``roboforge.data.processed`` applies it to ``states[t]`` of each demo), and
* closed-loop evaluation (Session 2's simulator adapter applies it after every ``env.step``).

Why not use robosuite's observations directly:

* robosuite v1.5.1 PickPlace evaluates ``Can_to_robot0_eef_pos/quat`` before ``Can_pos``/
  ``Can_quat``, so they read the previous update's cache (KI-003).
* After ``env.step`` robosuite's observables are read from kinematics computed *before* the
  last physics substep (robosuite calls ``sim.forward()``, the controller, then ``mj_step``,
  which integrates after computing kinematics). They lag the returned state by one 2 ms
  substep, while the robomimic v1.5 dataset observations were regenerated from saved states.

The function is pure: it copies ``qpos`` into a private scratch ``MjData`` and runs
``mj_kinematics`` there, so it never mutates the live simulation and gives the same value for
the same state whether the state came from a dataset file or a rollout. Feature definitions
and quaternion conventions are robosuite's. Every ``relation: equal`` feature reproduces
robosuite's observation bitwise wherever that observation is current. The corrected relative
position (``pos_in_eef_frame``) is evaluated in float64 from MuJoCo's rotation matrix, so it
agrees with robosuite's float32 implementation only to about 1e-7 m (KI-018).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from roboforge.data.specs import FeatureSpec, ObservationSpec


def robosuite_transforms() -> Any:
    """Import robosuite's transform utilities after the MuJoCo GL backend is configured."""
    from roboforge.sim.backend import configure_mujoco_gl

    configure_mujoco_gl()
    import robosuite.utils.transform_utils as T

    return T


class BoundFeatureExtractor:
    """An :class:`ObservationSpec` bound to one compiled MuJoCo model (IDs resolved by name)."""

    def __init__(self, spec: ObservationSpec, model: Any) -> None:
        import mujoco

        self._mj = mujoco
        self._T = robosuite_transforms()
        self.spec = spec
        self.model = model
        self._scratch = mujoco.MjData(model)
        self._resolve: dict[str, dict[str, Any]] = {f.name: self._ids(f) for f in spec.features}
        self._reject_mocap_dependence()

    def _reject_mocap_dependence(self) -> None:
        """Raise if any feature reads a body whose kinematic chain contains a mocap body.

        Mocap poses are not part of the flattened ``[time, qpos, qvel]`` state, so such a
        feature would not be a function of the state. robosuite 1.5.1's default models carry
        two non-colliding mocap markers (``*_eef_target``, KI-014); no feature depends on them,
        and the scratch data keeps their model-default pose.
        """
        m = self.model
        for name, ids in self._resolve.items():
            bodies = [int(m.site_bodyid[ids[k]]) for k in ("site", "ref_site") if k in ids]
            bodies += [ids[k] for k in ("body", "ref_body") if k in ids]
            bodies += [int(m.jnt_bodyid[j]) for j in range(m.njnt)
                       if int(m.jnt_qposadr[j]) in ids.get("qpos_adr", [])]
            for b in bodies:
                while b > 0:
                    if int(m.body_mocapid[b]) >= 0:
                        raise ValueError(f"feature {name!r} depends on a mocap body")
                    b = int(m.body_parentid[b])

    def _id(self, obj: Any, name: str | None) -> int:
        if name is None:
            raise ValueError("missing MuJoCo element name")
        idx = int(self._mj.mj_name2id(self.model, obj, name))
        if idx < 0:
            raise KeyError(f"MuJoCo element {name!r} ({obj}) not in model")
        return idx

    def _ids(self, f: FeatureSpec) -> dict[str, Any]:
        o = self._mj.mjtObj
        ids: dict[str, Any] = {}
        if f.site is not None:
            ids["site"] = self._id(o.mjOBJ_SITE, f.site)
        if f.body is not None:
            ids["body"] = self._id(o.mjOBJ_BODY, f.body)
        if f.ref_site is not None:
            ids["ref_site"] = self._id(o.mjOBJ_SITE, f.ref_site)
        if f.ref_body is not None:
            ids["ref_body"] = self._id(o.mjOBJ_BODY, f.ref_body)
        if f.joints:
            ids["qpos_adr"] = [int(self.model.jnt_qposadr[self._id(o.mjOBJ_JOINT, j)])
                               for j in f.joints]
        return ids

    def resolved_ids(self) -> dict[str, dict[str, Any]]:
        """MuJoCo IDs/qpos addresses per feature (for tests against robosuite's own IDs)."""
        return {k: dict(v) for k, v in self._resolve.items()}

    def _feature(self, f: FeatureSpec, d: Any) -> np.ndarray:
        T, ids = self._T, self._resolve[f.name]
        if f.kind == "site_pos":
            return np.array(d.site_xpos[ids["site"]])
        if f.kind == "body_pos":
            return np.array(d.xpos[ids["body"]])
        if f.kind == "body_quat_xyzw":
            return T.convert_quat(np.array(d.xquat[ids["body"]]), to="xyzw")
        if f.kind == "joint_qpos":
            return np.array([d.qpos[a] for a in ids["qpos_adr"]])
        if f.kind == "pos_diff_world":  # robosuite _get_obj_eef_sensor: obj_pos - eef_pos
            return np.array(d.xpos[ids["body"]]) - np.array(d.site_xpos[ids["ref_site"]])
        if f.kind == "pos_in_eef_frame":
            # robosuite _get_rel_obj_eef_sensor semantics, on same-step poses: the object
            # position in the frame with origin at the grip site and the orientation of the
            # hand body. Evaluated in float64 from MuJoCo's own rotation matrix: robosuite's
            # T.quat2mat casts to float32, which limits it to ~1e-7 and makes the value
            # platform-dependent under numba (KI-018), so this feature matches robosuite only
            # to float32 precision, by design.
            rot = np.array(d.xmat[ids["ref_body"]]).reshape(3, 3)
            return rot.T @ (np.array(d.xpos[ids["body"]]) - np.array(d.site_xpos[ids["ref_site"]]))
        raise ValueError(f"unknown feature kind {f.kind!r}")

    def from_flat_state(self, state: np.ndarray) -> np.ndarray:
        """Observation vector (float64, spec order) for one flattened MuJoCo state."""
        m, d = self.model, self._scratch
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (1 + m.nq + m.nv,):
            raise ValueError(f"state shape {state.shape} != ({1 + m.nq + m.nv},)")
        d.time = state[0]
        d.qpos[:] = state[1:1 + m.nq]
        d.qvel[:] = state[1 + m.nq:]
        self._mj.mj_kinematics(m, d)
        parts = [self._feature(f, d) for f in self.spec.features]
        for f, p in zip(self.spec.features, parts, strict=True):
            if p.shape != (f.dim,):
                raise AssertionError(f"{f.name}: computed shape {p.shape} != ({f.dim},)")
        return np.concatenate(parts)

    def from_states(self, states: np.ndarray) -> np.ndarray:
        """``[T, D]`` observation array for a ``[T, state_dim]`` state sequence."""
        return np.stack([self.from_flat_state(s) for s in states])

    def as_dict(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        """Split an observation vector into named feature blocks."""
        return {k: vector[lo:hi] for k, (lo, hi) in self.spec.slices().items()}


class FeatureExtractor:
    """Spec-level entry point; re-binds automatically when the simulator reloads its model."""

    def __init__(self, spec: ObservationSpec) -> None:
        self.spec = spec
        self._bound: BoundFeatureExtractor | None = None

    def bind(self, model: Any) -> BoundFeatureExtractor:
        """Return an extractor bound to ``model`` (a raw ``mujoco.MjModel``), cached by identity."""
        if self._bound is None or self._bound.model is not model:
            self._bound = BoundFeatureExtractor(self.spec, model)
        return self._bound

    def from_sim(self, sim: Any) -> np.ndarray:
        """Observation vector for the current state of a robosuite ``MjSim`` (no side effects)."""
        bound = self.bind(sim.model._model)
        return bound.from_flat_state(np.array(sim.get_state().flatten()))

    def dict_from_sim(self, sim: Any) -> dict[str, np.ndarray]:
        """Named feature blocks for the current state of a robosuite ``MjSim``."""
        return self.bind(sim.model._model).as_dict(self.from_sim(sim))
