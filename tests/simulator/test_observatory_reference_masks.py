"""Viewer-verification reference masks hide sites (D-091, KI-050).

MuJoCo's segmentation render draws sites whatever their alpha, so robosuite's transparent
``grip_site_cylinder`` (radius 5 mm, half-length 10 m) would cover robot pixels that the RGB
render shows. The reference masks must contain geoms only. Validation start, render only.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roboforge.sim.env_adapter import RoboForgeEnv
from roboforge.sim.eval_states import load_suite
from roboforge.sim.shadow_render import ShadowRenderer

pytestmark = [pytest.mark.sim, pytest.mark.render]


def test_reference_segmentation_has_no_site_pixels() -> None:
    init = load_suite("lift", "validation")[0]
    env = RoboForgeEnv("lift", horizon=4)
    try:
        env.reset(init)
        shadow = ShadowRenderer(env.model)
        try:
            state = env.current_state()
            site = int(mujoco.mjtObj.mjOBJ_SITE)
            with_sites = shadow.segmentation(state, "frontview", 240, 240)
            without = shadow.segmentation(state, "frontview", 240, 240, sites=False)
        finally:
            shadow.close()
    finally:
        env.close()
    assert np.any(with_sites[..., 1] == site)  # the default options do draw sites
    assert not np.any(without[..., 1] == site)
    geom = int(mujoco.mjtObj.mjOBJ_GEOM)
    assert (without[..., 1] == geom).sum() > (with_sites[..., 1] == geom).sum()
