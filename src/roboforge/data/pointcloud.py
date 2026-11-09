"""Metric depth and point clouds from simulator states (Session 9, Level 2; MASTER_SPEC §7.3).

This is the observation path of the DP3-inspired point-cloud policy. The dataset builder and
the closed-loop policy both call it, so one function of (compiled model, state) produces
every cloud.

**Depth.** :class:`DepthRenderer` renders on a scratch ``MjData`` that shares the compiled
model (never the live data; KI-019). It uses the D-074 visual options: geom groups
``(0, 1, 1, 0, 0, 0)``, robosuite's offscreen convention. It also switches **every site group
off**. robosuite hides its visualization sites in colour only, through a non-positive alpha,
and a hidden site still writes the depth buffer. Lift's grip-site sphere and its 10 m
grip-site cylinder would otherwise appear in depth as a phantom sphere and a line along the
gripper axis. E8-03 showed that turning the site groups off changes no RGB pixel, so this
is D-074's "every visualization site hidden" carried over to depth. Each new GL context first
renders once and discards the frame (``WARM_UP_RENDERS``, as for RGB).

MuJoCo 3.2.7's ``Renderer`` reads the depth buffer and converts it to metric depth:
* the buffer is reversed-Z, written with ``mjDEPTH_ZEROFAR``;
* ``near = vis.map.znear · stat.extent`` and ``far = vis.map.zfar · stat.extent``;
* metric depth is ``d_coef / (buffer + c_coef)`` in float64, cast to float32, where ``c_coef``
  and ``d_coef`` are the glFrustum coefficients after the reversed-Z transform.

The value is **z-depth**: the distance along the camera's viewing axis (−z), not along the
pixel's ray. :func:`depth_convention` records near, far and the convention per model. The
feasibility check compares the depth with exact ray casts (``mj_ray``) on every pixel.

**Back-projection.** A pixel ``(row i, column j)`` is sampled at its centre
``(u, v) = (j + 0.5, i + 0.5)``. With the pinhole intrinsics of
:meth:`roboforge.sim.kinematics.KinematicsExtractor.camera` (``f = (H/2) / tan(fovy/2)``,
principal point at the image centre, square pixels), the camera-frame point at z-depth ``d``
is ``((u − cx) d / fx, −(v − cy) d / fy, −d)``: x right, y up, z backward. The world point is
``R p + t``, with ``R = cam_xmat`` (camera axes as columns) and ``t = cam_xpos``. This is the
exact inverse of :func:`roboforge.data.rgb.project_world_point`.

**What the policy sees** (sensor-like; :class:`PointCloudSpec`):
* depth from two cameras, ``agentview`` and ``robot0_eye_in_hand``;
* each camera's calibration: intrinsics, and its world pose. The wrist camera's pose comes
  from robot kinematics, which a real robot knows from its joint encoders and a hand–eye
  calibration;
* a fixed workspace box in the world frame, a static configuration constant;
* deterministic farthest-point sampling to ``n_points``.

No segmentation, object pose or simulator label enters a cloud. :mod:`roboforge.data.pc_checks`
uses segmentation **only** to measure the clouds, never to build them.

The world frame is MuJoCo's world. Lift's robot base is fixed in it, so it is a fixed,
calibrated robot-workspace frame. Depth is noise-free and the calibration exact: both are
declared limitations.

Nothing here reads an evaluation suite.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SPEC_SCHEMA = "roboforge.point_cloud_spec/1"
WARM_UP_RENDERS = 1
# Pixels at or beyond this fraction of the far plane carry no surface (background).
FAR_FRACTION = 0.99


@dataclass(frozen=True)
class PointCloudSpec:
    """Everything that turns a state into a cloud; hashed into the dataset and checkpoints."""

    cameras: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
    size: int = 84
    box_lo: tuple[float, float, float] = (-0.25, -0.25, 0.805)
    box_hi: tuple[float, float, float] = (0.25, 0.25, 1.20)
    n_points: int = 512
    sampler: str = "fps_start0"
    dtype: str = "float32"
    schema: str = field(default=SPEC_SCHEMA)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PointCloudSpec:
        """Inverse of :meth:`to_dict`; refuses another schema or an unknown key."""
        d = dict(d)
        if d.get("schema", SPEC_SCHEMA) != SPEC_SCHEMA:
            raise ValueError(f"not a {SPEC_SCHEMA} spec: {d.get('schema')}")
        known = set(cls.__dataclass_fields__)
        extra = set(d) - known
        if extra:
            raise KeyError(f"unknown point-cloud spec keys {sorted(extra)}")
        for k in ("cameras", "box_lo", "box_hi"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON values."""
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}

    def sha256(self) -> str:
        """Hash of the canonical JSON form."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @property
    def centre(self) -> np.ndarray:
        """Centre of the workspace box (world, metres)."""
        return (np.asarray(self.box_lo, np.float64) + np.asarray(self.box_hi, np.float64)) / 2

    @property
    def half_extent(self) -> np.ndarray:
        """Half side lengths of the workspace box (metres)."""
        return (np.asarray(self.box_hi, np.float64) - np.asarray(self.box_lo, np.float64)) / 2


def depth_convention(model: Any) -> dict[str, Any]:
    """Near and far planes (metres) and the depth convention of ``model``'s renders."""
    extent = float(model.stat.extent)
    return {"znear": float(model.vis.map.znear), "zfar": float(model.vis.map.zfar),
            "extent": extent, "near_m": float(model.vis.map.znear) * extent,
            "far_m": float(model.vis.map.zfar) * extent,
            "buffer": "reversed-Z (mjDEPTH_ZEROFAR), converted by mujoco.Renderer",
            "value": "z-depth along the viewing axis (-z), metres, float32"}


class DepthRenderer:
    """Metric z-depth of flattened states for one compiled model (scratch data only)."""

    def __init__(self, model: Any) -> None:
        import mujoco

        from roboforge.sim.shadow_render import ROBOSUITE_GEOMGROUP

        self._mj = mujoco
        self.model = model
        self._data = mujoco.MjData(model)
        self._renderers: dict[int, Any] = {}
        self._seg: dict[int, Any] = {}
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = ROBOSUITE_GEOMGROUP
        self._opt.sitegroup[:] = 0  # hidden sites still write depth (module docstring)
        self.convention = depth_convention(model)

    def load(self, state: np.ndarray) -> Any:
        """Copy ``state`` into the scratch data and run ``mj_forward`` there."""
        m, d = self.model, self._data
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (1 + m.nq + m.nv,):
            raise ValueError(f"state shape {state.shape} != ({1 + m.nq + m.nv},)")
        d.time = state[0]
        d.qpos[:] = state[1:1 + m.nq]
        d.qvel[:] = state[1 + m.nq:]
        self._mj.mj_forward(m, d)
        return d

    def _renderer(self, size: int) -> Any:
        if size not in self._renderers:
            r = self._mj.Renderer(self.model, height=size, width=size)
            r.enable_depth_rendering()
            self._renderers[size] = r
            for _ in range(WARM_UP_RENDERS):  # discard the cold first render of the context
                r.update_scene(self._data, camera=0, scene_option=self._opt)
                r.render()
        return self._renderers[size]

    def depth(self, state: np.ndarray, camera: str, size: int) -> np.ndarray:
        """``float32 [size, size]`` z-depth in metres, upright (row 0 = top)."""
        d = self.load(state)
        r = self._renderer(size)
        r.update_scene(d, camera=camera, scene_option=self._opt)
        out = np.ascontiguousarray(r.render())
        if out.dtype != np.float32 or out.shape != (size, size):
            raise ValueError(f"unexpected depth {out.dtype} {out.shape} from {camera}")
        return out

    def segmentation(self, state: np.ndarray, camera: str, size: int) -> np.ndarray:
        """**Diagnostics only**: ``int32 [size, size, 2]`` (object id, type), same options."""
        d = self.load(state)
        if size not in self._seg:
            r = self._mj.Renderer(self.model, height=size, width=size)
            r.enable_segmentation_rendering()
            self._seg[size] = r
        r = self._seg[size]
        r.update_scene(d, camera=camera, scene_option=self._opt)
        return np.ascontiguousarray(r.render())

    def close(self) -> None:
        """Release the GL contexts."""
        for r in [*self._renderers.values(), *self._seg.values()]:
            r.close()
        self._renderers.clear()
        self._seg.clear()


def pixel_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Continuous ``(u, v)`` of every pixel centre, each ``[size · size]`` in row-major order."""
    c = np.arange(size, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(c, c)  # uu[i, j] = j + 0.5, vv[i, j] = i + 0.5
    return uu.reshape(-1), vv.reshape(-1)


def backproject(camera: dict[str, Any], u: np.ndarray, v: np.ndarray,
                depth: np.ndarray) -> np.ndarray:
    """World points ``[n, 3]`` (float64) of image coordinates ``(u, v)`` at z-depth ``depth``."""
    k = camera["intrinsics"]
    d = np.asarray(depth, dtype=np.float64)
    p_cam = np.stack([(np.asarray(u) - k["cx"]) * d / k["fx"],
                      -(np.asarray(v) - k["cy"]) * d / k["fy"], -d], axis=-1)
    rot = np.asarray(camera["rot_world_from_cam"], dtype=np.float64)
    return p_cam @ rot.T + np.asarray(camera["pos"], dtype=np.float64)


def depth_to_world(camera: dict[str, Any], depth: np.ndarray,
                   far_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Every pixel of a depth image as world points ``[H·W, 3]`` and its validity mask.

    A pixel is valid when its depth is finite, positive and below ``FAR_FRACTION · far``."""
    size = depth.shape[0]
    u, v = pixel_grid(size)
    dep = depth.reshape(-1).astype(np.float64)
    valid = np.isfinite(dep) & (dep > 0) & (dep < FAR_FRACTION * far_m)
    return backproject(camera, u, v, dep), valid


def in_box(points: np.ndarray, spec: PointCloudSpec) -> np.ndarray:
    """Boolean mask of points inside the closed workspace box."""
    lo, hi = np.asarray(spec.box_lo), np.asarray(spec.box_hi)
    return np.all((points >= lo) & (points <= hi), axis=-1)


def farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Indices of ``n`` points chosen by farthest-point sampling, deterministically.

    Starts at index 0. Each step takes the point farthest (squared Euclidean distance,
    float64, summed x + y + z in that order) from the chosen set; ``argmax`` ties go to the
    lowest index. With fewer than ``n`` points, the full FPS order repeats cyclically to fill
    ``n`` (max pooling is unchanged by duplicates). An empty input raises."""
    m = int(points.shape[0])
    if m == 0:
        raise ValueError("empty point set: nothing inside the workspace box")
    p = np.asarray(points, dtype=np.float64)
    k = min(n, m)
    idx = np.empty(k, dtype=np.int64)
    idx[0] = 0
    diff = p - p[0]
    dist = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2]
    for i in range(1, k):
        j = int(np.argmax(dist))
        idx[i] = j
        diff = p - p[j]
        dist = np.minimum(dist, diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
                          + diff[:, 2] * diff[:, 2])
    if k < n:
        idx = np.resize(idx, n)
    return idx


def cloud_from_depths(depths: dict[str, np.ndarray], cameras: dict[str, dict[str, Any]],
                      spec: PointCloudSpec, far_m: float) -> tuple[np.ndarray, dict[str, Any]]:
    """The policy's cloud ``[n_points, 3]`` (``spec.dtype``, world metres) and its counts.

    Points of every camera (in ``spec.cameras`` order, each row-major) are pooled, kept if
    valid and inside the box, then reduced by :func:`farthest_point_sample`."""
    pts, src = [], []
    info: dict[str, Any] = {"n_valid": {}, "n_in_box": {}}
    for ci, cam in enumerate(spec.cameras):
        w, valid = depth_to_world(cameras[cam], depths[cam], far_m)
        keep = valid & in_box(w, spec)
        info["n_valid"][cam] = int(valid.sum())
        info["n_in_box"][cam] = int(keep.sum())
        pts.append(w[keep])
        src.append(np.full(int(keep.sum()), ci, dtype=np.int8))
    allp = np.concatenate(pts)
    idx = farthest_point_sample(allp, spec.n_points)
    info["n_crop"] = int(allp.shape[0])
    info["n_unique"] = int(min(spec.n_points, allp.shape[0]))
    info["source_camera"] = np.concatenate(src)[idx]
    return allp[idx].astype(spec.dtype), info


class CloudMaker:
    """State → cloud for one compiled model: depth renders, calibration, crop and sampling."""

    def __init__(self, model: Any, spec: PointCloudSpec) -> None:
        from roboforge.sim.kinematics import KinematicsExtractor

        self.model, self.spec = model, spec
        self.depth = DepthRenderer(model)
        self.kin = KinematicsExtractor(model)

    def cameras(self, state: np.ndarray) -> dict[str, dict[str, Any]]:
        """Each camera's world pose and intrinsics at ``state`` (the wrist camera moves)."""
        s = self.spec.size
        return {c: self.kin.camera(state, c, s, s) for c in self.spec.cameras}

    def depths(self, state: np.ndarray) -> dict[str, np.ndarray]:
        """Each camera's z-depth image at ``state``."""
        return {c: self.depth.depth(state, c, self.spec.size) for c in self.spec.cameras}

    def cloud(self, state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """The policy's cloud at ``state``."""
        return cloud_from_depths(self.depths(state), self.cameras(state), self.spec,
                                 self.depth.convention["far_m"])

    def close(self) -> None:
        """Release the GL contexts."""
        self.depth.close()


def cloud_sha256(cloud: np.ndarray) -> str:
    """SHA-256 of (dtype, shape, bytes), the frame-hash convention."""
    from roboforge.data.rgb import frame_sha256

    return frame_sha256(cloud)


__all__ = ["FAR_FRACTION", "SPEC_SCHEMA", "WARM_UP_RENDERS", "CloudMaker", "DepthRenderer",
           "PointCloudSpec", "backproject", "cloud_from_depths", "cloud_sha256",
           "depth_convention", "depth_to_world", "farthest_point_sample", "in_box",
           "pixel_grid"]
