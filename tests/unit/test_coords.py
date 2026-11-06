"""MuJoCo → three.js conversion and projection math (no simulator needed)."""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.viz import coords


def _rand_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_basis_is_proper_rotation_z_up_to_y_up() -> None:
    assert np.allclose(coords.C @ coords.C.T, np.eye(3))
    assert np.isclose(np.linalg.det(coords.C), 1.0)
    np.testing.assert_array_equal(coords.mj_to_three_pos(np.array([1.0, 2.0, 3.0])),
                                  [1.0, 3.0, -2.0])
    np.testing.assert_allclose(coords.quat_to_mat(coords.Q_C_WXYZ), coords.C, atol=1e-15)


def test_quaternion_conversion_matches_matrix_conversion() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = _rand_quat(rng)
        x, y, z, w = coords.mj_to_three_quat_xyzw(q)
        np.testing.assert_allclose(coords.quat_to_mat(np.array([w, x, y, z])),
                                   coords.C @ coords.quat_to_mat(q), atol=1e-14)


def test_body_point_transforms_consistently() -> None:
    """A body-local point lands at C(R p + t) under the converted pose."""
    rng = np.random.default_rng(1)
    q, t, p = _rand_quat(rng), rng.normal(size=3), rng.normal(size=3)
    world_mj = coords.quat_to_mat(q) @ p + t
    x, y, z, w = coords.mj_to_three_quat_xyzw(q)
    world_three = coords.quat_to_mat(np.array([w, x, y, z])) @ p + coords.mj_to_three_pos(t)
    np.testing.assert_allclose(world_three, coords.mj_to_three_pos(world_mj), atol=1e-14)


@pytest.mark.parametrize("size", [(256, 256), (320, 240)])
def test_three_projection_equals_mujoco_projection(size: tuple[int, int]) -> None:
    w, h = size
    rng = np.random.default_rng(2)
    q = _rand_quat(rng)
    rot = coords.quat_to_mat(q)
    cam_pos = rng.normal(size=3)
    pts = cam_pos - (rot[:, 2] * rng.uniform(1, 3, size=(20, 1))) + rng.normal(0, 0.2, (20, 3))
    px_mj = coords.project_mujoco(pts, cam_pos, rot, 45.0, w, h)
    px_3 = coords.project_three(coords.mj_to_three_pos(pts), coords.mj_to_three_pos(cam_pos),
                                coords.mj_to_three_quat_xyzw(q), 45.0, w, h)
    np.testing.assert_allclose(px_3, px_mj, atol=1e-9)


def test_point_on_optical_axis_projects_to_image_centre() -> None:
    rot = np.eye(3)
    px = coords.project_mujoco(np.array([[0.0, 0.0, -2.0]]), np.zeros(3), rot, 60.0, 200, 100)
    np.testing.assert_allclose(px, [[100.0, 50.0]])


def test_forward_kinematics_hinge_and_slide() -> None:
    hier = [{"name": "a", "parent": None, "local_pos": [0, 0, 1], "local_quat_wxyz": [1, 0, 0, 0],
             "joints": [{"name": "h", "type": "hinge", "axis": [0, 0, 1], "pos": [1, 0, 0],
                         "qpos0": [0.0]}]},
            {"name": "b", "parent": "a", "local_pos": [2, 0, 0], "local_quat_wxyz": [1, 0, 0, 0],
             "joints": [{"name": "s", "type": "slide", "axis": [1, 0, 0], "pos": [0, 0, 0],
                         "qpos0": [0.0]}]}]
    out = coords.forward_kinematics(hier, {"h": [np.pi / 2], "s": [0.5]})
    # hinge about z through (1, 0, 1): body origin (0, 0, 1) rotates to (1, -1, 1)
    np.testing.assert_allclose(out["a"][0], [1.0, -1.0, 1.0], atol=1e-12)
    # child at local (2, 0, 0) + slide 0.5 along the rotated x axis (now +y)
    np.testing.assert_allclose(out["b"][0], [1.0, 1.5, 1.0], atol=1e-12)
