"""Pure-math helpers of the replay checker and MuJoCo backend selection."""

from __future__ import annotations

import os

import numpy as np
import pytest

from roboforge.sim.backend import default_backend
from roboforge.sim.state_replay import StateLayout, quat_angle, state_errors


def test_quat_angle_identity_and_antipode() -> None:
    q = np.array([0.9, 0.1, -0.3, 0.2])
    assert quat_angle(q, q) == pytest.approx(0.0, abs=1e-6)
    assert quat_angle(q, -q) == pytest.approx(0.0, abs=1e-6)  # q and -q are the same rotation


def test_quat_angle_known_rotation() -> None:
    ident = np.array([1.0, 0.0, 0.0, 0.0])  # MuJoCo scalar-first
    rz90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    assert quat_angle(ident, rz90) == pytest.approx(np.pi / 2, abs=1e-9)


def test_state_errors_decomposition() -> None:
    # nq = 2 robot + 7 free joint = 9 ; nv = 2 + 6 = 8 ; state = 1 + 9 + 8 = 18
    layout = StateLayout(nq=9, nv=8, robot_qpos_idx=[1, 2], robot_qvel_idx=[10, 11],
                         object_pos_idx=[[3, 4, 5]], object_quat_idx=[[6, 7, 8, 9]],
                         object_names=["cube"])
    expected = np.zeros(18)
    expected[6] = 1.0
    actual = expected.copy()
    actual[2] += 0.01      # robot joint
    actual[4] -= 0.002     # object y
    actual[15] += 3.0      # an object angular velocity
    e = state_errors(layout, actual, expected)
    assert e["robot_qpos_max_abs_rad"] == pytest.approx(0.01)
    assert e["object_pos_max_abs_m"] == pytest.approx(0.002)
    assert e["object_rot_max_rad"] == pytest.approx(0.0)
    assert e["qvel_max_abs"] == pytest.approx(3.0)
    assert layout.state_dim == 18


def test_default_backend_per_platform() -> None:
    assert default_backend("darwin") == "cgl"
    assert default_backend("linux") == "egl"
    with pytest.raises(RuntimeError):
        default_backend("win32")


def test_backend_configured_before_mujoco_import() -> None:
    assert os.environ["MUJOCO_GL"] in {"cgl", "glfw", "egl", "osmesa", "glx"}
