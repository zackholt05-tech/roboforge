"""MuJoCo world → three.js world conversion and camera projection (DECISIONS D-030).

**Frames.** MuJoCo's world frame is right-handed, z up, in metres. three.js is right-handed,
y up. The conversion is the proper rotation of −90° about x:

    p_three = C p_mj,   C = [[1, 0, 0], [0, 0, 1], [0, −1, 0]],   i.e. (x, y, z) → (x, z, −y)

A body's orientation maps as ``R_three = C R_mj``. Its local geometry is *not* re-expressed,
so meshes and primitives stay in MuJoCo body coordinates. In quaternions:
``q_three = q_C ⊗ q_mj`` with ``q_C = (cos(−π/4), sin(−π/4), 0, 0)`` (w, x, y, z). three.js
stores quaternions as ``(x, y, z, w)``; MuJoCo as ``(w, x, y, z)``.

A viewer can equivalently put every MuJoCo-frame object under one root ``Group`` with
``rotation.x = −π/2`` and use MuJoCo positions and reordered quaternions verbatim inside it.
``tests/web_contract`` checks that this matches :func:`mj_to_three_pos` with the real
three.js library.

**Cameras.** MuJoCo cameras look along their local −z with local +y up, the same convention
as ``THREE.PerspectiveCamera``, so a camera converts like any body.
``fov = cam_fovy`` (vertical, degrees) and ``aspect = width / height``.
:func:`project_mujoco` projects in MuJoCo's frame. :func:`project_three` is a **Python
re-implementation** of three.js's ``Vector3.project`` (``makePerspective`` + view matrix),
labelled as such; the real three.js library is exercised in ``tests/web_contract``. Pixel
coordinates are continuous, origin at the top-left image corner, so pixel ``(i, j)`` covers
``[j, j + 1) × [i, i + 1)``.
"""

from __future__ import annotations

import numpy as np

C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
Q_C_WXYZ = np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0, 0.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of ``(w, x, y, z)`` quaternions (broadcasts over leading dims)."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit ``(w, x, y, z)`` quaternion."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(r: np.ndarray) -> np.ndarray:
    """Unit ``(w, x, y, z)`` quaternion of a rotation matrix (w ≥ 0)."""
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(r, dtype=np.float64).ravel())
    return q if q[0] >= 0 else -q


def mj_to_three_pos(p: np.ndarray) -> np.ndarray:
    """MuJoCo world positions ``[..., 3]`` → three.js world positions."""
    return np.asarray(p, dtype=np.float64) @ C.T


def mj_to_three_quat_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo world orientations ``(w, x, y, z)`` → three.js world quaternions ``(x, y, z, w)``."""
    q = quat_mul(Q_C_WXYZ, q_wxyz)
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def project_mujoco(points: np.ndarray, cam_pos: np.ndarray, rot_world_from_cam: np.ndarray,
                   fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Pinhole projection of MuJoCo world points ``[N, 3]`` into continuous pixel coords."""
    pc = (np.asarray(points, dtype=np.float64) - cam_pos) @ rot_world_from_cam
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)
    x_ndc = (f * height / width) * pc[:, 0] / -pc[:, 2]
    y_ndc = f * pc[:, 1] / -pc[:, 2]
    return np.stack([(x_ndc + 1) / 2 * width, (1 - y_ndc) / 2 * height], axis=1)


def three_camera_matrices(position: np.ndarray, quat_xyzw: np.ndarray, fov_deg: float,
                          aspect: float, near: float = 0.01,
                          far: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """(projectionMatrix, matrixWorldInverse) as three.js computes them (re-implementation)."""
    x, y, z, w = quat_xyzw
    world = np.eye(4)
    world[:3, :3] = quat_to_mat(np.array([w, x, y, z]))
    world[:3, 3] = position
    top = near * np.tan(np.deg2rad(fov_deg) / 2.0)
    height = 2 * top
    width = aspect * height
    left, right, bottom = -width / 2, width / 2, top - height
    proj = np.zeros((4, 4))
    proj[0, 0] = 2 * near / (right - left)
    proj[1, 1] = 2 * near / (top - bottom)
    proj[0, 2] = (right + left) / (right - left)
    proj[1, 2] = (top + bottom) / (top - bottom)
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj, np.linalg.inv(world)


def project_three(points_three: np.ndarray, position: np.ndarray, quat_xyzw: np.ndarray,
                  fov_deg: float, width: int, height: int) -> np.ndarray:
    """three.js ``Vector3.project`` + NDC→pixel, re-implemented in NumPy (see module doc)."""
    proj, view = three_camera_matrices(position, quat_xyzw, fov_deg, width / height)
    ph = np.concatenate([np.asarray(points_three, dtype=np.float64),
                         np.ones((len(points_three), 1))], axis=1)
    clip = ph @ (proj @ view).T
    ndc = clip[:, :3] / clip[:, 3:4]
    return np.stack([(ndc[:, 0] + 1) / 2 * width, (1 - ndc[:, 1]) / 2 * height], axis=1)


def forward_kinematics(hierarchy: list[dict], qpos_by_joint: dict[str, list[float]]
                       ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World (pos, quat_wxyz) of every body from the exported hierarchy + joint values.

    This is the algorithm a joint-angle-driven viewer implements. Each body's local rest
    transform is composed onto its parent's world transform, then its joints apply in order:
    a hinge rotates by ``q − qpos0`` about ``axis`` through ``pos``; a slide translates by
    ``q − qpos0`` along ``axis``; a free joint sets the pose. All are expressed in the body
    frame, following MuJoCo's ``mj_kinematics``.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for b in hierarchy:
        pp, pq = (out[b["parent"]] if b["parent"] else (np.zeros(3), np.array([1.0, 0, 0, 0])))
        pos = pp + quat_to_mat(pq) @ np.asarray(b["local_pos"])
        quat = quat_mul(pq, np.asarray(b["local_quat_wxyz"]))
        for j in b["joints"]:
            q = np.asarray(qpos_by_joint[j["name"]], dtype=np.float64)
            if j["type"] == "free":
                pos, quat = q[:3].copy(), q[3:7] / np.linalg.norm(q[3:7])
                continue
            axis, anchor = np.asarray(j["axis"]), np.asarray(j["pos"])
            delta = float(q[0] - j["qpos0"][0])
            if j["type"] == "hinge":
                xanchor = pos + quat_to_mat(quat) @ anchor
                local = np.concatenate([[np.cos(delta / 2)], np.sin(delta / 2) * axis])
                quat = quat_mul(quat, local)
                pos = xanchor - quat_to_mat(quat) @ anchor
            elif j["type"] == "slide":
                pos = pos + quat_to_mat(quat) @ axis * delta
            else:
                raise NotImplementedError(f"joint type {j['type']}")
        out[b["name"]] = (pos, quat / np.linalg.norm(quat))
    return out
