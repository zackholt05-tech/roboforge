"""Frame-indexed robot, object and camera kinematics as a pure function of the MuJoCo state.

Like :mod:`roboforge.data.features`, everything here is computed on a private scratch
``MjData``: the flattened state ``[time, qpos, qvel]`` is copied in and ``mj_kinematics``,
``mj_comPos`` and ``mj_camlight`` run there. The live simulation is never touched. This
matters: after ``env.step`` the live kinematics lag the state by one 2 ms physics substep
(KI-015), and ``mj_forward`` on the live data would overwrite MuJoCo's solver warm start
and change the rest of the rollout (DECISIONS D-024).

Conventions (MuJoCo, documented in ``docs/SIMULATOR_CONTRACT.md``):

* world frame: right-handed, z up, metres;
* body poses: ``xpos`` (frame origin, world) and ``xquat`` (**w, x, y, z**, scalar first);
* camera frame: x right, y up, looking along **−z**; vertical field of view ``cam_fovy``.

Bodies are resolved **by name** for each compiled model: robosuite's default models carry two
mocap markers that the dataset models lack (KI-014), so body IDs differ between them. Mocap
bodies and the world body are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ROBOT_ROOT_BODY = "robot0_base"


@dataclass(frozen=True)
class BodyInfo:
    """Static description of one exported body (for a hierarchy-aware viewer)."""

    name: str
    parent: str | None
    category: str  # "robot" | "free_object" | "fixture"
    local_pos: tuple[float, ...]
    local_quat_wxyz: tuple[float, ...]
    joints: tuple[dict[str, Any], ...]


class KinematicsExtractor:
    """Kinematics of one compiled ``mujoco.MjModel`` from flattened states (no side effects)."""

    def __init__(self, model: Any) -> None:
        import mujoco

        self._mj = mujoco
        self.model = model
        self._scratch = mujoco.MjData(model)
        m = model
        name = lambda obj, i: mujoco.mj_id2name(m, obj, i)  # noqa: E731
        ob = mujoco.mjtObj
        root = mujoco.mj_name2id(m, ob.mjOBJ_BODY, ROBOT_ROOT_BODY)
        if root < 0:
            raise KeyError(f"robot root body {ROBOT_ROOT_BODY!r} not in model")
        self.body_ids = [b for b in range(1, m.nbody) if int(m.body_mocapid[b]) < 0]
        self.body_names = [name(ob.mjOBJ_BODY, b) for b in self.body_ids]
        free = {int(m.jnt_bodyid[j]) for j in range(m.njnt)
                if int(m.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE}
        self._category = {b: ("robot" if self._descends(b, root) else
                              "free_object" if b in free else "fixture") for b in self.body_ids}
        self.joint_ids = [j for j in range(m.njnt)
                          if int(m.jnt_type[j]) != mujoco.mjtJoint.mjJNT_FREE]
        self.joint_names = [name(ob.mjOBJ_JOINT, j) for j in self.joint_ids]
        self._qadr = np.array([int(m.jnt_qposadr[j]) for j in self.joint_ids], dtype=int)
        self._vadr = np.array([int(m.jnt_dofadr[j]) for j in self.joint_ids], dtype=int)
        self.camera_names = [name(ob.mjOBJ_CAMERA, c) for c in range(m.ncam)]

    @property
    def body_categories(self) -> list[str]:
        """Category of each exported body, aligned with :attr:`body_names`."""
        return [self._category[b] for b in self.body_ids]

    def _descends(self, b: int, root: int) -> bool:
        while b > 0:
            if b == root:
                return True
            b = int(self.model.body_parentid[b])
        return False

    def _load(self, state: np.ndarray) -> Any:
        m, d = self.model, self._scratch
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (1 + m.nq + m.nv,):
            raise ValueError(f"state shape {state.shape} != ({1 + m.nq + m.nv},)")
        d.time = state[0]
        d.qpos[:] = state[1:1 + m.nq]
        d.qvel[:] = state[1 + m.nq:]
        self._mj.mj_kinematics(m, d)
        self._mj.mj_comPos(m, d)
        self._mj.mj_camlight(m, d)
        return d

    def frame(self, state: np.ndarray) -> dict[str, np.ndarray]:
        """Per-frame arrays: joint qpos/qvel and world pose of every exported body."""
        d = self._load(state)
        ids = self.body_ids
        return {"joint_qpos": np.array(d.qpos[self._qadr]),
                "joint_qvel": np.array(d.qvel[self._vadr]),
                "body_pos": np.array(d.xpos[ids]),
                "body_quat_wxyz": np.array(d.xquat[ids])}

    def robot(self, state: np.ndarray) -> dict[str, Any]:
        """Joint state and world transforms of the robot links (incl. gripper and mount)."""
        f = self.frame(state)
        keep = [i for i, b in enumerate(self.body_ids) if self._category[b] == "robot"]
        return {"joint_names": list(self.joint_names), "joint_qpos": f["joint_qpos"],
                "joint_qvel": f["joint_qvel"],
                "link_names": [self.body_names[i] for i in keep],
                "link_pos": f["body_pos"][keep], "link_quat_wxyz": f["body_quat_wxyz"][keep]}

    def scene(self, state: np.ndarray) -> dict[str, Any]:
        """World transforms of the non-robot bodies (free objects and fixtures)."""
        f = self.frame(state)
        keep = [i for i, b in enumerate(self.body_ids) if self._category[b] != "robot"]
        return {"body_names": [self.body_names[i] for i in keep],
                "categories": [self._category[self.body_ids[i]] for i in keep],
                "body_pos": f["body_pos"][keep], "body_quat_wxyz": f["body_quat_wxyz"][keep]}

    def site_pos(self, state: np.ndarray, site: str) -> np.ndarray:
        """World position of a named site."""
        sid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_SITE, site)
        if sid < 0:
            raise KeyError(f"site {site!r} not in model")
        return np.array(self._load(state).site_xpos[sid])

    def camera(self, state: np.ndarray, camera: str, width: int, height: int) -> dict[str, Any]:
        """Camera pose (world), vertical FOV and pinhole intrinsics for a ``width×height`` image.

        ``rot_world_from_cam`` has the camera axes as columns (x right, y up, z backward).
        Intrinsics follow from MuJoCo's vertical FOV with square pixels and the principal point
        at the image centre: ``fy = fx = (height / 2) / tan(fovy / 2)``.
        """
        cid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_CAMERA, camera)
        if cid < 0:
            raise KeyError(f"camera {camera!r} not in model")
        d = self._load(state)
        fovy = float(self.model.cam_fovy[cid])
        f = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        return {"name": camera, "pos": np.array(d.cam_xpos[cid]),
                "rot_world_from_cam": np.array(d.cam_xmat[cid]).reshape(3, 3),
                "fovy_deg": fovy, "width": int(width), "height": int(height),
                "intrinsics": {"fx": float(f), "fy": float(f), "cx": width / 2.0,
                               "cy": height / 2.0}}

    def hierarchy(self) -> list[BodyInfo]:
        """Static body tree with local rest transforms and joint definitions (model frame)."""
        mj, m = self._mj, self.model
        out = []
        for b, nm in zip(self.body_ids, self.body_names, strict=True):
            p = int(m.body_parentid[b])
            joints = []
            for j in range(int(m.body_jntadr[b]), int(m.body_jntadr[b]) + int(m.body_jntnum[b])):
                joints.append({"name": mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j),
                               "type": {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[
                                   int(m.jnt_type[j])],
                               "axis": [float(x) for x in m.jnt_axis[j]],
                               "pos": [float(x) for x in m.jnt_pos[j]],
                               "qpos0": [float(x) for x in m.qpos0[int(m.jnt_qposadr[j]):
                                         int(m.jnt_qposadr[j]) + (7 if int(m.jnt_type[j]) == 0
                                                                   else 1)]]})
            out.append(BodyInfo(
                name=nm, parent=None if p == 0 else mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, p),
                category=self._category[b],
                local_pos=tuple(float(x) for x in m.body_pos[b]),
                local_quat_wxyz=tuple(float(x) for x in m.body_quat[b]),
                joints=tuple(joints)))
        return out
