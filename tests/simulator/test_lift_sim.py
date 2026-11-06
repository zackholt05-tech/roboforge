"""Simulator contract tests for Lift (no dataset, no network; needs robosuite + MuJoCo)."""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.sim.factory import load_sim_config, make_env
from roboforge.sim.state_replay import is_success, reset_to_state, state_layout

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def env():
    e = make_env(load_sim_config("lift"))
    yield e
    e.close()


def _seeded_state(env, seed: int) -> np.ndarray:
    np.random.seed(seed)
    env.reset()
    return env.sim.get_state().flatten().copy()


def test_action_space_matches_dataset_controller(env) -> None:
    low, high = env.action_spec
    assert env.action_dim == 7
    assert np.all(low == -1.0) and np.all(high == 1.0)
    assert env.control_freq == 20 and env.control_timestep == pytest.approx(0.05)


def test_state_layout_lift(env) -> None:
    layout = state_layout(env)
    assert layout.object_names == ["cube_joint0"]
    assert layout.state_dim == 32 and len(layout.robot_qpos_idx) == 9  # 7 arm + 2 finger joints


def test_global_seed_controls_reset(env) -> None:
    assert np.array_equal(_seeded_state(env, 3), _seeded_state(env, 3))
    assert not np.array_equal(_seeded_state(env, 3), _seeded_state(env, 4))


def test_step_advances_time_and_stays_finite(env) -> None:
    _seeded_state(env, 0)
    t0 = env.sim.data.time
    rng = np.random.default_rng(0)
    for _ in range(10):
        obs, reward, done, info = env.step(rng.uniform(-1, 1, env.action_dim))
    assert env.sim.data.time - t0 == pytest.approx(10 * env.control_timestep)
    assert all(np.all(np.isfinite(v)) for v in obs.values() if isinstance(v, np.ndarray))
    assert reward == 0.0 and not is_success(env)


def test_success_predicate_responds_to_cube_height(env) -> None:
    _seeded_state(env, 0)
    assert not is_success(env)
    qz = state_layout(env).object_pos_idx[0][2] - 1  # flattened-state index -> qpos index
    env.sim.data.qpos[qz] += 0.10
    env.sim.forward()
    assert is_success(env)


def test_state_restore_round_trip_is_exact(env) -> None:
    target = _seeded_state(env, 11)
    xml = env.sim.model.get_xml()
    _seeded_state(env, 12)  # move elsewhere first
    restored = reset_to_state(env, xml, target)
    assert np.array_equal(restored, target)


@pytest.mark.parametrize(("offset_m", "expected"), [(0.039, False), (0.041, True)])
def test_success_threshold_boundary(env, offset_m: float, expected: bool) -> None:
    """Lift success is cube height > table top + 0.04 m (robosuite Lift._check_success)."""
    _seeded_state(env, 0)
    qz = state_layout(env).object_pos_idx[0][2] - 1
    env.sim.data.qpos[qz] = env.model.mujoco_arena.table_offset[2] + offset_m
    env.sim.forward()
    assert is_success(env) is expected


@pytest.mark.parametrize("dim", [0, 6, 8])
def test_wrong_action_dimension_is_rejected(env, dim: int) -> None:
    _seeded_state(env, 0)
    with pytest.raises((AssertionError, ValueError)):
        env.step(np.zeros(dim))


def test_restore_rejects_wrong_state_length(env) -> None:
    xml = env.sim.model.get_xml()
    with pytest.raises((AssertionError, ValueError)):
        reset_to_state(env, xml, np.zeros(10))
