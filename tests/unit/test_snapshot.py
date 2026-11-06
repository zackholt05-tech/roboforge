"""Snapshot serialization: portable XML, hashing, partitions (no simulator needed)."""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.eval.policies import RandomPolicy, episode_seed
from roboforge.sim.snapshot import InitialState, portable_xml


def test_portable_xml_rewrites_only_asset_paths() -> None:
    xml = ('<mujoco><asset><mesh file="/home/u/code/robosuite-dev/robosuite/models/assets/'
           'robots/panda/meshes/link0.stl"/><texture file="/Users/x/.venv/lib/python3.11/'
           'site-packages/robosuite/models/assets/arenas/../textures/a.png"/></asset>'
           '<light dir="-0.19 -0.19 -0.96"/></mujoco>')
    out = portable_xml(xml)
    assert 'file="robosuite/models/assets/robots/panda/meshes/link0.stl"' in out
    assert 'file="robosuite/models/assets/arenas/../textures/a.png"' in out
    assert 'dir="-0.19 -0.19 -0.96"' in out
    assert "/home/" not in out and "/Users/" not in out


def test_portable_xml_rejects_unknown_absolute_paths() -> None:
    with pytest.raises(ValueError):
        portable_xml('<mujoco><mesh file="/opt/other/mesh.stl"/></mujoco>')
    with pytest.raises(ValueError):
        portable_xml('<mujoco><texture fileup="/abs/robosuite/x.png"/></mujoco>')


def test_initial_state_round_trip_and_tamper_detection() -> None:
    init = InitialState("lift", "lift_x", "eval_validation", "<mujoco/>",
                        np.array([0.0, 0.1, -2.5e-17]), {"seed": 1})
    again = InitialState.from_dict(init.to_dict())
    assert np.array_equal(again.state, init.state)
    assert again.state_sha256 == init.state_sha256
    bad = init.to_dict()
    bad["state"][1] += 1e-16
    with pytest.raises(ValueError, match="state hash"):
        InitialState.from_dict(bad)
    bad = init.to_dict()
    bad["model_xml"] += " "
    with pytest.raises(ValueError, match="XML hash"):
        InitialState.from_dict(bad)


def test_partition_and_state_validation() -> None:
    with pytest.raises(ValueError):
        InitialState("lift", "x", "test", "<mujoco/>", np.zeros(3))
    with pytest.raises(ValueError):
        InitialState("lift", "x", "eval_validation", "<mujoco/>", np.array([np.nan]))


def test_random_policy_episode_streams_are_reproducible_and_distinct() -> None:
    from roboforge.data.pipeline import load_specs

    _, act = load_specs("lift")

    class Ep:  # minimal stand-in for EpisodeMetadata
        def __init__(self, sid: str) -> None:
            self.initial_state = {"state_id": sid}

    a, b = RandomPolicy(act, 0), RandomPolicy(act, 0)
    a.reset_episode(Ep("s0"))
    b.reset_episode(Ep("s0"))
    xa = np.concatenate([a.act(None) for _ in range(5)])
    xb = np.concatenate([b.act(None) for _ in range(5)])
    np.testing.assert_array_equal(xa, xb)
    assert xa.shape == (5, act.dim) and np.all(np.abs(xa) <= 1.0)
    b.reset_episode(Ep("s1"))
    assert not np.array_equal(xa[0], b.act(None)[0])
    assert episode_seed(0, "s0") != episode_seed(1, "s0")
