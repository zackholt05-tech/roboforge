"""The failure taxonomy's Can goal-bin footprint equals robosuite's (D-065)."""

from __future__ import annotations

import numpy as np

from roboforge.eval import taxonomy as tx


def test_goal_bin_half_size_matches_robosuite() -> None:
    from roboforge.sim.factory import load_sim_config, make_env

    env = make_env(load_sim_config("can"))
    try:
        half = np.asarray(env.bin_size[:2]) / 2
    finally:
        env.close()
    assert np.allclose(half, tx.taxonomy_config()["can_goal_bin"]["half_size_xy_m"])
