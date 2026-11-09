"""Level 2 point-cloud geometry (Session 9): projection round trip, sampling, crop, encoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from roboforge.data.pointcloud import (
    PointCloudSpec,
    backproject,
    cloud_from_depths,
    farthest_point_sample,
    in_box,
    pixel_grid,
)
from roboforge.data.rgb import project_world_point
from roboforge.models.point_encoder import DP3Encoder, PointCloudObservationEncoder


def _camera(seed: int, size: int = 84, fovy: float = 45.0) -> dict:
    """A random camera pose (rotation from a random quaternion) with RoboForge intrinsics."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    f = (size / 2) / np.tan(np.deg2rad(fovy) / 2)
    return {"pos": rng.normal(size=3), "rot_world_from_cam": rot,
            "intrinsics": {"fx": f, "fy": f, "cx": size / 2, "cy": size / 2}}


@pytest.mark.parametrize("seed", range(5))
def test_world_to_pixel_to_world_round_trip(seed: int) -> None:
    cam = _camera(seed)
    rng = np.random.default_rng(100 + seed)
    for _ in range(50):
        p_cam = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), -rng.uniform(0.05, 3)])
        p = np.asarray(cam["rot_world_from_cam"]) @ p_cam + cam["pos"]
        u, v, z = project_world_point(cam, p)
        back = backproject(cam, np.array([u]), np.array([v]), np.array([z]))[0]
        assert z == pytest.approx(-p_cam[2], rel=1e-12)
        np.testing.assert_allclose(back, p, rtol=0, atol=1e-12)


def test_pixel_centres_back_project_onto_their_own_pixels() -> None:
    cam = _camera(7, size=16)
    u, v = pixel_grid(16)
    assert (u[:3].tolist(), v[:3].tolist()) == ([0.5, 1.5, 2.5], [0.5, 0.5, 0.5])
    pts = backproject(cam, u, v, np.full(u.size, 0.7))
    for n in (0, 17, 255):
        uu, vv, z = project_world_point(cam, pts[n])
        assert (uu, vv, z) == pytest.approx((u[n], v[n], 0.7), abs=1e-12)


def test_fps_is_deterministic_and_farthest_first() -> None:
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0]])
    assert farthest_point_sample(pts, 4).tolist() == [0, 3, 2, 1]
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(2000, 3))
    a, b = farthest_point_sample(cloud, 512), farthest_point_sample(cloud.copy(), 512)
    assert np.array_equal(a, b) and a[0] == 0 and len(set(a.tolist())) == 512


def test_fps_pads_cyclically_and_refuses_an_empty_set() -> None:
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [3, 0, 0]])
    assert farthest_point_sample(pts, 7).tolist() == [0, 2, 1, 0, 2, 1, 0]
    with pytest.raises(ValueError, match="empty"):
        farthest_point_sample(np.zeros((0, 3)), 4)


def test_box_is_closed_and_spec_round_trips() -> None:
    spec = PointCloudSpec()
    pts = np.array([spec.box_lo, spec.box_hi, np.add(spec.box_hi, [0, 0, 1e-9])])
    assert in_box(pts, spec).tolist() == [True, True, False]
    assert PointCloudSpec.from_dict(spec.to_dict()) == spec
    assert PointCloudSpec.from_dict(spec.to_dict()).sha256() == spec.sha256()
    with pytest.raises(KeyError):
        PointCloudSpec.from_dict({**spec.to_dict(), "colour": True})


def test_cloud_of_a_flat_floor_lies_on_the_floor() -> None:
    """A camera 1 m above z = 0.9 looking straight down sees a plane at depth 1 m."""
    size = 32
    rot = np.eye(3)  # camera axes = world axes: it looks along its −z, straight down
    f = (size / 2) / np.tan(np.deg2rad(45) / 2)
    cam = {"pos": np.array([0.0, 0.0, 1.9]), "rot_world_from_cam": rot,
           "intrinsics": {"fx": f, "fy": f, "cx": size / 2, "cy": size / 2}}
    spec = PointCloudSpec(cameras=("a",), size=size, box_lo=(-1, -1, 0.85),
                          box_hi=(1, 1, 0.95), n_points=64)
    cloud, info = cloud_from_depths({"a": np.full((size, size), 1.0, np.float32)}, {"a": cam},
                                    spec, far_m=500.0)
    assert cloud.shape == (64, 3) and cloud.dtype == np.float32
    np.testing.assert_allclose(cloud[:, 2], 0.9, atol=1e-6)
    assert info["n_crop"] == size * size and info["n_unique"] == 64
    far = {"a": np.full((size, size), 499.0, np.float32)}  # beyond 0.99 · far: background
    with pytest.raises(ValueError, match="empty"):
        cloud_from_depths(far, {"a": cam}, spec, far_m=500.0)


def test_dp3_encoder_is_permutation_invariant_and_box_normalized() -> None:
    torch.manual_seed(0)
    enc = DP3Encoder()
    x = torch.randn(3, 100, 3)
    perm = torch.randperm(100)
    torch.testing.assert_close(enc(x), enc(x[:, perm]))
    assert enc(x).shape == (3, 64)
    obs = PointCloudObservationEncoder(9, (0.0, 0.0, 1.0), (0.25, 0.25, 0.2))
    pts = torch.tensor([[0.25, -0.25, 1.2]]).expand(2, 1, 5, 3)
    out = obs(pts, torch.zeros(2, 1, 9))
    assert out.shape == (2, 1, 73) and obs.out_dim == 73
    torch.testing.assert_close(obs.box_centre + obs.box_half, torch.tensor([0.25, 0.25, 1.2]))
