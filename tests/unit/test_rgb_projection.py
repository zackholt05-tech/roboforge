"""Pinhole projection, mask statistics and frame hashing of ``roboforge.data.rgb``."""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.data.rgb import frame_sha256, mask_stats, project_world_point
from roboforge.data.rgb_checks import dilate


def camera(size: int = 84, fovy: float = 45.0, pos=(0.0, 0.0, 0.0), rot=None) -> dict:
    f = (size / 2) / np.tan(np.deg2rad(fovy) / 2)
    return {"pos": np.array(pos, dtype=float),
            "rot_world_from_cam": np.eye(3) if rot is None else np.asarray(rot, dtype=float),
            "intrinsics": {"fx": f, "fy": f, "cx": size / 2, "cy": size / 2}}


def test_point_on_the_optical_axis_projects_to_the_image_centre() -> None:
    u, v, depth = project_world_point(camera(), np.array([0.0, 0.0, -2.0]))
    assert (u, v, depth) == (42.0, 42.0, 2.0)


def test_x_right_and_y_up_with_row_zero_at_the_top() -> None:
    cam = camera()
    f = cam["intrinsics"]["fx"]
    u, v, _ = project_world_point(cam, np.array([0.1, 0.0, -1.0]))
    assert u == pytest.approx(42 + 0.1 * f) and v == pytest.approx(42)
    u, v, _ = project_world_point(cam, np.array([0.0, 0.1, -1.0]))
    assert u == pytest.approx(42) and v == pytest.approx(42 - 0.1 * f)  # up = smaller row


def test_camera_pose_is_applied_and_points_behind_have_no_pixel() -> None:
    # Camera at (1, 0, 0) looking along world +x: camera z (backward) = world -x, y = world z.
    rot = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cam = camera(pos=(1.0, 0.0, 0.0), rot=rot)
    u, v, depth = project_world_point(cam, np.array([3.0, 0.0, 0.0]))
    assert (u, v) == pytest.approx((42.0, 42.0)) and depth == pytest.approx(2.0)
    u, v, depth = project_world_point(cam, np.array([0.0, 0.0, 0.0]))
    assert np.isnan(u) and np.isnan(v) and depth < 0


def test_mask_stats_centroid_on_mask_and_distance() -> None:
    m = np.zeros((10, 10), dtype=bool)
    m[2:4, 6:8] = True  # pixel centres x in {6.5, 7.5}, y in {2.5, 3.5}
    s = mask_stats(m, 7.0, 3.0)
    assert s["area_px"] == 4 and s["centroid_uv"] == [7.0, 3.0]
    assert s["projection_on_mask"] and s["dist_to_centroid_px"] == 0.0
    far = mask_stats(m, 1.0, 9.0)
    assert not far["projection_on_mask"] and far["dist_to_mask_px"] > 5
    empty = mask_stats(np.zeros((4, 4), dtype=bool), 1.0, 1.0)
    assert empty["area_px"] == 0 and empty["centroid_uv"] is None


def test_frame_hash_covers_shape_dtype_and_bytes() -> None:
    a = np.zeros((2, 3, 3), dtype=np.uint8)
    assert frame_sha256(a) == frame_sha256(a.copy())
    assert frame_sha256(a) != frame_sha256(a.reshape(3, 2, 3))
    b = a.copy()
    b[0, 0, 0] = 1
    assert frame_sha256(a) != frame_sha256(b)


def test_dilate_square_neighbourhood_and_borders() -> None:
    m = np.zeros((7, 7), dtype=bool)
    m[3, 3] = True
    assert dilate(m, 1).sum() == 9 and dilate(m, 2).sum() == 25
    corner = np.zeros((5, 5), dtype=bool)
    corner[0, 0] = True
    assert np.argwhere(dilate(corner, 1)).tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
