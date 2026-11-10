"""Observatory media drawn from recorded states (D-074 / D-083 paths; DECISIONS D-089).

* :func:`render_video`: ``frontview`` frames of every recorded state by
  :class:`~roboforge.sim.shadow_render.ShadowRenderer` (scratch data, robosuite's visual geom
  groups), one discarded warm-up render per GL context, written as H.264 MP4 at the 20 Hz
  control rate. It is the viewer's inset and its reduced-motion / no-WebGL fallback.
* :func:`reference_frames`: MuJoCo RGB and a **silhouette class mask** (0 background, 1 robot,
  2 task object, 3 other visible geoms) at listed frames, with each camera's pose and field of
  view. The browser renders the same frames from the same cameras and the two silhouettes are
  compared (``make web-verify``).
* :func:`policy_clouds`: the Level 2 policy's 512-point world-frame cloud at each decision
  frame, recomputed from the recorded state by the frozen :class:`CloudMaker` (D-083).
* :func:`policy_rgb_frames`: the RGB arm's two 84² camera images at each decision frame.

Rendering is a pure function of the recorded state, so none of this can change a result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from roboforge.viz import coords

VIDEO_CAMERA, VIDEO_SIZE, VIDEO_FPS = "frontview", 256, 20
REF_CAMERAS, REF_SIZE = ("frontview", "agentview"), 480
CLASS_BACKGROUND, CLASS_ROBOT, CLASS_OBJECT, CLASS_OTHER = 0, 1, 2, 3


def _warm(shadow: Any, state: np.ndarray, camera: str, size: int) -> None:
    shadow.rgb(state, camera, size, size)  # discarded warm-up render (D-074)


def render_video(shadow: Any, states: np.ndarray, path: Path) -> dict[str, Any]:
    """Write the frontview MP4 of ``states``; return its description."""
    import imageio.v2 as iio

    _warm(shadow, states[0], VIDEO_CAMERA, VIDEO_SIZE)
    frames = [shadow.rgb(s, VIDEO_CAMERA, VIDEO_SIZE, VIDEO_SIZE) for s in states]
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(path, frames, fps=VIDEO_FPS, macro_block_size=8, quality=7)
    return {"camera": VIDEO_CAMERA, "size": VIDEO_SIZE, "fps": VIDEO_FPS,
            "n_frames": len(frames), "codec": "h264 (imageio-ffmpeg)",
            "label": "MuJoCo offscreen render of the recorded states (simulator ground truth)"}


def class_mask(model: Any, seg: np.ndarray, robot_bodies: set[int],
               object_bodies: set[int]) -> np.ndarray:
    """``uint8`` silhouette classes of a MuJoCo segmentation render (geoms only)."""
    import mujoco

    is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    gid = np.where(is_geom, seg[..., 0], -1)
    body = np.where(gid >= 0, np.asarray(model.geom_bodyid)[np.clip(gid, 0, None)], -1)
    out = np.zeros(seg.shape[:2], dtype=np.uint8)
    out[gid >= 0] = CLASS_OTHER
    out[np.isin(body, list(robot_bodies))] = CLASS_ROBOT
    out[np.isin(body, list(object_bodies))] = CLASS_OBJECT
    return out


def reference_frames(shadow: Any, kin: Any, model: Any, states: np.ndarray,
                     frames: list[int], out_dir: Path, stem: str,
                     categories: dict[str, str]) -> list[dict[str, Any]]:
    """MuJoCo RGB + class-mask PNGs at ``frames`` for :data:`REF_CAMERAS`; their metadata."""
    import imageio.v3 as iio
    import mujoco

    ob = mujoco.mjtObj
    ids = {n: mujoco.mj_name2id(model, ob.mjOBJ_BODY, n) for n in categories}
    robot = {ids[n] for n, c in categories.items() if c == "robot"}
    obj = {ids[n] for n, c in categories.items() if c == "free_object"}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for cam_name in REF_CAMERAS:
        _warm(shadow, states[0], cam_name, REF_SIZE)
    for k in frames:
        for cam_name in REF_CAMERAS:
            s = states[k]
            rgb = shadow.rgb(s, cam_name, REF_SIZE, REF_SIZE)
            seg = shadow.segmentation(s, cam_name, REF_SIZE, REF_SIZE, sites=False)  # D-091
            mask = class_mask(model, seg, robot, obj)
            cam = kin.camera(s, cam_name, REF_SIZE, REF_SIZE)
            rgb_p = out_dir / f"{stem}_t{k:03d}_{cam_name}_rgb.png"
            mask_p = out_dir / f"{stem}_t{k:03d}_{cam_name}_mask.png"
            iio.imwrite(rgb_p, rgb)
            iio.imwrite(mask_p, mask)
            out.append({"frame": k, "camera": cam_name, "size": REF_SIZE,
                        "rgb": rgb_p.name, "mask": mask_p.name,
                        "mask_classes": {"0": "background", "1": "robot", "2": "task object",
                                         "3": "other visual geom"},
                        "mask_render": "MuJoCo segmentation, robosuite visual geom groups, "
                                       "sites hidden (D-091)",
                        "pixels": {"robot": int((mask == CLASS_ROBOT).sum()),
                                   "object": int((mask == CLASS_OBJECT).sum())},
                        "camera_pose": {"pos": [float(x) for x in cam["pos"]],
                                        "quat_wxyz": [float(x) for x in coords.mat_to_quat(
                                            cam["rot_world_from_cam"])],
                                        "fovy_deg": float(cam["fovy_deg"])}})
    return out


def decision_frame_indices(arrays: dict[str, np.ndarray]) -> list[int]:
    """Frames at which the policy decided (action ``t`` with ``chunk_offset == 0``)."""
    off = arrays["chunk_offset"]
    return [int(t) for t in np.flatnonzero(off == 0)]


def policy_clouds(model: Any, spec: Any, states: np.ndarray, frames: list[int]
                  ) -> dict[str, Any]:
    """The Level 2 policy's cloud at each decision frame (float32 world metres → 4 decimals)."""
    from roboforge.data.pointcloud import CloudMaker, cloud_sha256

    maker = CloudMaker(model, spec)
    try:
        clouds, src, hashes, info = [], [], [], []
        for k in frames:
            c, inf = maker.cloud(states[k])
            clouds.append(np.round(c.astype(np.float64), 4).tolist())
            src.append([int(x) for x in inf["source_camera"]])
            hashes.append(cloud_sha256(c))
            info.append({"n_crop": inf["n_crop"], "n_valid": inf["n_valid"],
                         "n_in_box": inf["n_in_box"]})
    finally:
        maker.close()
    return {"frames": frames, "points": clouds, "source_camera": src,
            "cameras": list(spec.cameras), "cloud_sha256": hashes, "info": info,
            "spec": spec.to_dict(), "spec_sha256": spec.sha256(),
            "label": "the policy's input cloud at each decision frame, recomputed from the "
                     "recorded state by the frozen D-083 path (depth + calibration only)"}


def policy_rgb_frames(shadow: Any, states: np.ndarray, frames: list[int], out_dir: Path,
                      stem: str, cameras: tuple[str, ...] = ("agentview", "robot0_eye_in_hand"),
                      size: int = 84) -> dict[str, Any]:
    """The RGB arm's two 84² camera images at each decision frame, as PNG files."""
    import imageio.v3 as iio

    from roboforge.data.rgb import frame_sha256

    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cameras:
        _warm(shadow, states[0], c, size)
    files: dict[str, list[str]] = {c: [] for c in cameras}
    hashes: dict[str, list[str]] = {c: [] for c in cameras}
    for k in frames:
        for c in cameras:
            img = shadow.rgb(states[k], c, size, size)
            p = out_dir / f"{stem}_t{k:03d}_{c}.png"
            iio.imwrite(p, img)
            files[c].append(p.name)
            hashes[c].append(frame_sha256(img))
    return {"frames": frames, "cameras": list(cameras), "size": size, "files": files,
            "frame_sha256": hashes,
            "label": "the RGB policy's two 84² camera inputs at each decision frame, "
                     "re-rendered from the recorded state by the D-074 path"}
