"""Level 2 depth and point clouds are calibrated and consistent (Session 9, MASTER_SPEC §7.3).

On a real Lift demonstration state:
* rendered z-depth equals exact ray casts on every interior pixel;
* a known simulator point (the cube's top-face centre) projects to a pixel and back within the
  pixel's footprint, onto its plane;
* back-projected cube and table pixels lie on their true surfaces, for both cameras;
* the cloud has no hidden-site phantoms, is deterministic, and is equal for a live and a restored
  state;
* the sampled cloud covers the cube.

Segmentation, poses and ray casts appear only in these checks, never in a cloud.
"""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.data import pc_checks
from roboforge.data.pointcloud import CloudMaker
from roboforge.data.rgb import DemoFrameRenderer, iter_raw_demos, raw_file
from roboforge.train.pointcloud import PCConfig, config_path

pytestmark = [pytest.mark.sim, pytest.mark.data, pytest.mark.render]


@pytest.fixture(scope="module")
def setup():
    try:
        raw_file("lift", "raw")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    demo = next(iter_raw_demos("lift", ["demo_0"]))
    r = DemoFrameRenderer("lift")
    r.load_model(demo.model_xml, demo.states[0])
    mk = CloudMaker(r.model, PCConfig.load(config_path()).cloud_spec)
    yield demo, r, mk
    mk.close()
    r.close()


@pytest.mark.parametrize("camera", ["agentview", "robot0_eye_in_hand"])
def test_depth_equals_ray_casts_on_interior_pixels(setup, camera: str) -> None:
    demo, _, mk = setup
    res = pc_checks.depth_vs_ray(mk, demo.states[0], camera)
    assert res["ray_hits"] == res["pixels"]
    assert res["interior_pixels"] > 3000 and res["interior_pixels_over_1mm"] == 0
    assert res["interior_abs_err_max_m"] < 1e-4


@pytest.mark.parametrize("size", [84, 256])
def test_known_point_projects_to_pixel_and_back(setup, size: int) -> None:
    demo, _, mk = setup
    res = pc_checks.known_point_check(mk, demo.states[0], "agentview", size)
    assert res["applicable"], res
    assert res["plane_residual_m"] < 1e-4
    assert res["reproj_err_px"] < 1e-9
    assert res["within_footprint"] and res["dist_to_known_m"] <= res["footprint_bound_m"]


def test_back_projected_surfaces_lie_on_cube_and_table(setup) -> None:
    demo, _, mk = setup
    cube = pc_checks.cube_surface_residual(mk, demo.states[0], 84)
    table = pc_checks.table_plane_residual(mk, demo.states[0], 84)
    for cam in ("agentview", "robot0_eye_in_hand"):
        assert cube[cam]["n"] > 10 and cube[cam]["abs_max_m"] < 1e-4
        assert table[cam]["n"] > 1000 and table[cam]["abs_max_m"] < 1e-4


def test_hidden_sites_are_out_of_the_depth_path(setup) -> None:
    demo, _, mk = setup
    res = pc_checks.sites_in_depth(mk, demo.states[0], "robot0_eye_in_hand", 84)
    assert res["pixels_changed_by_sites"] > 0  # they would write depth if left on
    seg = mk.depth.segmentation(demo.states[0], "robot0_eye_in_hand", 84)
    import mujoco

    assert not np.any(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_SITE))


def test_cloud_is_deterministic_live_equals_restored_and_covers_the_cube(setup) -> None:
    demo, _, mk = setup
    c1, info = mk.cloud(demo.states[0])
    c2, _ = mk.cloud(demo.states[0])
    assert np.array_equal(c1, c2) and c1.shape == (512, 3) and c1.dtype == np.float32
    assert info["n_crop"] >= 512
    cov = pc_checks.cube_coverage(mk, demo.states[0])
    assert cov["on_cube_sampled"] >= 10
    b = DemoFrameRenderer("lift")
    try:
        b.load_model(demo.model_xml, demo.states[0])
        for act in demo.actions[:10]:
            b.env.step(np.asarray(act, dtype=np.float64))
        live = np.array(b.env.sim.get_state().flatten())
        mk_live = CloudMaker(b.model, mk.spec)
        live_cloud, _ = mk_live.cloud(live)
        mk_live.close()
    finally:
        b.close()
    assert np.array_equal(live_cloud, mk.cloud(live)[0])
