"""RGB frames rendered from recorded demonstration states (Session 8, MASTER_SPEC §7.2).

The robomimic v1.5 raw files (``demo_v15.hdf5``) hold, per demonstration, the flattened MuJoCo
states ``[time, qpos, qvel]`` and the saved model XML (``model_file``) but **no images**. A
camera frame for step ``t`` is therefore rendered from ``states[t]`` itself:

1. the demo's ``model_file`` is compiled through the same hard reset that every RoboForge
   restore uses (:func:`roboforge.sim.state_replay.reset_to_state`). robosuite's reset toggles
   every visualization site off (``visualize`` with all settings ``False``), so no site whose
   colour robosuite derives from privileged state (Lift colours the gripper site by its
   distance to the cube) is visible;
2. :class:`roboforge.sim.shadow_render.ShadowRenderer` copies ``states[t]`` into a scratch
   ``MjData`` and renders there. Closed-loop evaluation renders the live state through the same
   class (``RoboForgeEnv.render``), so training and evaluation frames come from one function
   of (compiled model, state, camera, size) and the live rollout is never touched (KI-019).

Frame ``t``, the low-dim observation ``obs[t]`` (``FeatureExtractor`` of ``states[t]``) and the
action ``actions[t]`` share the index ``t``: the frame shows the state from which the
demonstrator chose ``actions[t]``. Every frame carries an alignment record (schema
:data:`FRAME_SCHEMA`): task, demo, step index, the SHA-256 of the state (float64 little-endian
bytes, as :func:`roboforge.sim.snapshot.sha256_state`) and of the model XML string, the camera
name with its pinhole intrinsics and world extrinsics, the GL backend and the frame's SHA-256.

Conventions (``docs/SIMULATOR_CONTRACT.md``): images are upright ``uint8 [H, W, 3]`` (row 0 =
top). A MuJoCo camera looks along its local −z with y up; ``cam_xmat`` holds the camera axes as
columns. Pixel ``(row i, column j)`` covers ``[j, j+1) × [i, i+1)`` in continuous image
coordinates ``(u, v)``, the principal point is the image centre ``(W/2, H/2)`` and
``f = (H/2) / tan(fovy/2)`` with square pixels.

Nothing here reads an evaluation suite: inputs are demonstration states only.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

FRAME_SCHEMA = "roboforge.rgb_frame/1"
# robomimic's image setups (generate_paper_configs.py at d309eae; dataset_states_to_obs.py
# examples): an external camera and a wrist camera at 84 x 84.
DEFAULT_CAMERAS = ("agentview", "robot0_eye_in_hand")
# The task object whose projected position the alignment tests compare with its visible
# location (the free body of configs/simulator/<task>.yaml task_object_joints).
TASK_OBJECT_BODY = {"lift": "cube_main", "can": "Can_main"}
# The low-dim observation slice holding that body's world position (data/manifests/tasks).
TASK_OBJECT_POS_FEATURE = {"lift": "cube_pos", "can": "Can_pos"}
# Discarded renders per fresh GL context before any frame is kept (see DemoFrameRenderer.rgb).
WARM_UP_RENDERS = 1


def frame_sha256(frame: np.ndarray) -> str:
    """SHA-256 of (dtype, shape, bytes) of one image, the ``arrays_sha256`` convention."""
    a = np.ascontiguousarray(frame)
    h = hashlib.sha256(f"{a.dtype.str}|{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def project_world_point(camera: dict[str, Any], point: np.ndarray) -> tuple[float, float, float]:
    """Continuous image coordinates ``(u, v)`` and forward depth of a world point.

    ``camera`` is :meth:`roboforge.sim.kinematics.KinematicsExtractor.camera` output (``pos``,
    ``rot_world_from_cam`` with columns x right, y up, z backward, and ``intrinsics``). The
    depth is the distance along the viewing direction (−z); it is ≤ 0 behind the camera.
    """
    rot = np.asarray(camera["rot_world_from_cam"], dtype=np.float64)
    p_cam = rot.T @ (np.asarray(point, dtype=np.float64) - np.asarray(camera["pos"]))
    depth = float(-p_cam[2])
    k = camera["intrinsics"]
    if depth <= 0:
        return float("nan"), float("nan"), depth
    u = k["cx"] + k["fx"] * p_cam[0] / depth
    v = k["cy"] - k["fy"] * p_cam[1] / depth
    return float(u), float(v), depth


def body_geom_ids(model: Any, body_name: str) -> np.ndarray:
    """Geom IDs of ``body_name`` and every body below it (visual and collision geoms)."""
    import mujoco

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if root < 0:
        raise KeyError(f"body {body_name!r} not in model")
    bodies = {root}
    for b in range(model.nbody):  # parents precede children in MuJoCo's body order
        if int(model.body_parentid[b]) in bodies:
            bodies.add(b)
    return np.array([g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies],
                    dtype=np.int64)


def object_mask(segmentation: np.ndarray, geom_ids: np.ndarray) -> np.ndarray:
    """Boolean ``[H, W]`` mask of pixels whose segmentation shows one of ``geom_ids``."""
    import mujoco

    obj, typ = segmentation[..., 0], segmentation[..., 1]
    return (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(obj, geom_ids)


def mask_stats(mask: np.ndarray, u: float, v: float) -> dict[str, Any]:
    """Where the projected point ``(u, v)`` falls relative to a visible-object mask.

    Returns the mask area, its centroid in continuous coordinates (pixel centres at
    ``j + 0.5, i + 0.5``), the distance from the projection to that centroid, whether the
    pixel containing the projection is part of the mask, and its distance (pixels) to the
    nearest mask pixel centre.
    """
    rows, cols = np.nonzero(mask)
    area = int(rows.size)
    h, w = mask.shape
    inside_image = bool(np.isfinite(u) and np.isfinite(v) and 0 <= u < w and 0 <= v < h)
    out: dict[str, Any] = {"area_px": area, "projection_uv": [u, v],
                           "projection_in_image": inside_image}
    if area == 0:
        return {**out, "centroid_uv": None, "dist_to_centroid_px": None,
                "projection_on_mask": False, "dist_to_mask_px": None}
    cu, cv = float(cols.mean() + 0.5), float(rows.mean() + 0.5)
    on_mask = bool(inside_image and mask[int(v), int(u)])
    if np.isfinite(u) and np.isfinite(v):
        dist_mask = float(np.min(np.hypot(cols + 0.5 - u, rows + 0.5 - v)))
        dist_c = float(np.hypot(cu - u, cv - v))
    else:
        dist_mask = dist_c = float("nan")
    return {**out, "centroid_uv": [cu, cv], "dist_to_centroid_px": dist_c,
            "projection_on_mask": on_mask, "dist_to_mask_px": dist_mask}


@dataclass(frozen=True)
class RawDemo:
    """One demonstration of the raw file: its saved model XML, states and actions."""

    task: str
    demo: str
    model_xml: str
    states: np.ndarray
    actions: np.ndarray


def raw_file(task: str, kind: str = "raw") -> tuple[Any, str]:
    """``(path, sha256)`` of a task's downloaded PH file, verified against the manifest."""
    from roboforge.data.fetch import load_manifest, verify_file

    _, files = load_manifest()
    entry = next(f for f in files if f.task == task and f.dataset_type == "ph"
                 and f.hdf5_type == kind)
    path = entry.destination()
    if not path.exists():
        raise FileNotFoundError(f"{path} missing ({task}/{kind}); run `make download`")
    return path, verify_file(entry, path)


def iter_raw_demos(task: str, demos: list[str] | None = None) -> Iterator[RawDemo]:
    """Demonstrations of the hash-verified raw file, in numeric order (or ``demos`` order)."""
    import h5py

    from roboforge.data.inspect import demo_keys

    path, _ = raw_file(task, "raw")
    with h5py.File(path, "r") as f:
        for demo in demos or demo_keys(f):
            g = f["data"][demo]
            yield RawDemo(task=task, demo=demo, model_xml=str(g.attrs["model_file"]),
                          states=g["states"][()], actions=g["actions"][()])


def jsonable_camera(camera: dict[str, Any]) -> dict[str, Any]:
    """Camera pose and intrinsics as plain JSON values."""
    return {"name": camera["name"], "width": camera["width"], "height": camera["height"],
            "fovy_deg": camera["fovy_deg"], "intrinsics": dict(camera["intrinsics"]),
            "pos_world": [float(x) for x in camera["pos"]],
            "rot_world_from_cam": [[float(x) for x in row]
                                   for row in np.asarray(camera["rot_world_from_cam"])]}


class DemoFrameRenderer:
    """Render recorded states of one task's demonstrations (scratch data only).

    The MuJoCo GL backend must be configured before this module's caller imports mujoco
    (:func:`roboforge.sim.backend.configure_mujoco_gl`).
    """

    def __init__(self, task: str) -> None:
        from roboforge.sim.factory import load_sim_config, make_env

        self.task = task
        self.env = make_env(load_sim_config(task))
        self._xml: str | None = None
        self.model_xml_sha256: str | None = None
        self._shadow: Any = None
        self._kin: Any = None
        self._object_geoms: np.ndarray | None = None
        self._warm: set[int] = set()

    def load_model(self, model_xml: str, state0: np.ndarray) -> None:
        """Compile ``model_xml`` through the RoboForge hard reset (no-op if already loaded)."""
        from roboforge.sim.kinematics import KinematicsExtractor
        from roboforge.sim.shadow_render import ShadowRenderer
        from roboforge.sim.snapshot import sha256_text
        from roboforge.sim.state_replay import reset_to_state

        if model_xml == self._xml:
            return
        back = reset_to_state(self.env, model_xml, np.asarray(state0, dtype=np.float64))
        if not np.array_equal(back, state0):
            raise AssertionError(f"{self.task}: restore read-back differs from the state")
        if self._shadow is not None:
            self._shadow.close()
        model = self.env.sim.model._model
        self._shadow, self._kin = ShadowRenderer(model), KinematicsExtractor(model)
        self._warm = set()
        self._object_geoms = body_geom_ids(model, TASK_OBJECT_BODY[self.task])
        self._xml, self.model_xml_sha256 = model_xml, sha256_text(model_xml)

    @property
    def model(self) -> Any:
        """The compiled model of the loaded demo."""
        return self.env.sim.model._model

    @property
    def object_geoms(self) -> np.ndarray:
        """Geom IDs of the task object in the loaded model."""
        if self._object_geoms is None:
            raise RuntimeError("load_model() first")
        return self._object_geoms

    def rgb(self, state: np.ndarray, camera: str, size: int, *, warm: bool = True) -> np.ndarray:
        """Upright ``uint8 [size, size, 3]`` frame of ``state``.

        The first render of a fresh GL context can differ from every later render of the same
        state by a few pixels (measured natively with CGL, Session 8). With ``warm`` (the
        default, and the only mode used for data), each context first renders ``state`` once
        and discards it (:data:`WARM_UP_RENDERS`), so every returned frame is a steady-state
        frame. ``warm=False`` exists only to measure that artifact.
        """
        if warm and size not in self._warm:
            for _ in range(WARM_UP_RENDERS):
                self._shadow.rgb(state, camera, size, size)
            self._warm.add(size)
        return self._shadow.rgb(state, camera, size, size)

    def segmentation(self, state: np.ndarray, camera: str, size: int) -> np.ndarray:
        """``int32 [size, size, 2]`` (object id, object type) per pixel; −1 = background."""
        return self._shadow.segmentation(state, camera, size, size)

    def camera(self, state: np.ndarray, camera: str, size: int) -> dict[str, Any]:
        """Camera pose (world) and intrinsics at ``state`` (the wrist camera moves)."""
        return self._kin.camera(state, camera, size, size)

    def object_position(self, state: np.ndarray) -> np.ndarray:
        """World position of the task object's body frame at ``state`` (scratch kinematics)."""
        scene = self._kin.scene(state)
        return np.asarray(scene["body_pos"][scene["body_names"].index(
            TASK_OBJECT_BODY[self.task])])

    def frame_record(self, demo: str, step_index: int, state: np.ndarray, camera: str,
                     size: int, frame: np.ndarray) -> dict[str, Any]:
        """The alignment record of one rendered frame."""
        from roboforge.sim.snapshot import sha256_state

        return {"schema": FRAME_SCHEMA, "task": self.task, "demo": demo,
                "step_index": int(step_index), "state_sha256": sha256_state(state),
                "model_xml_sha256": self.model_xml_sha256,
                "camera": jsonable_camera(self.camera(state, camera, size)),
                "backend": os.environ.get("MUJOCO_GL"), "frame_sha256": frame_sha256(frame),
                "frame_shape": list(frame.shape), "frame_dtype": str(frame.dtype)}

    def close(self) -> None:
        """Release GL contexts and the simulator."""
        if self._shadow is not None:
            self._shadow.close()
            self._shadow = None
        self.env.close()
