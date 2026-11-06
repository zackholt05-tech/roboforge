"""The canonical feature function in closed loop (no dataset needed; specs from the manifest).

These tests pin down the train/eval link of D-010. After any ``env.step``, the features of
the live simulation must equal the features of the same flattened state restored into a
fresh simulator. Computing them must not perturb the rollout. With current kinematics and a
fresh observation cache, robosuite's observations must equal them. Raw ``env.step``
observations do not (stale kinematics); that is why policies never consume them.
"""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.data.features import FeatureExtractor
from roboforge.data.pipeline import load_specs
from roboforge.sim.factory import load_sim_config, make_env
from roboforge.sim.state_replay import reset_to_state

pytestmark = pytest.mark.sim
N_STEPS = 25


def _actions(seed: int, dim: int) -> np.ndarray:
    a = np.random.default_rng(seed).uniform(-1, 1, size=(N_STEPS, dim))
    a[:, -1] = np.where(np.arange(N_STEPS) > N_STEPS // 2, 1.0, -1.0)
    return a


def _start(task: str):
    env = make_env(load_sim_config(task))
    np.random.seed(0)
    env.reset()
    return env, env.sim.model.get_xml(), np.array(env.sim.get_state().flatten())


def _robosuite_key(f) -> str:
    return "object-state" if f.dataset_ref.key == "object" else f.dataset_ref.key


@pytest.mark.parametrize("task", ["lift", "can"])
def test_features_after_step_equal_features_of_restored_state(task: str) -> None:
    spec, act_spec = load_specs(task)
    env, xml, s0 = _start(task)
    # Both sims compile the same XML string (get_xml() re-serializes with rounded numbers, so
    # the originally generated model differs from it at ~1e-7).
    reset_to_state(env, xml, s0)
    other = make_env(load_sim_config(task))
    reset_to_state(other, xml, s0)
    live, restored = FeatureExtractor(spec), FeatureExtractor(spec)
    for a in _actions(1, act_spec.dim):
        env.step(a)
        state = np.array(env.sim.get_state().flatten())
        other.sim.set_state_from_flattened(state)
        other.sim.forward()
        np.testing.assert_array_equal(live.from_sim(env.sim), restored.from_sim(other.sim))
    env.close()
    other.close()


@pytest.mark.parametrize("task", ["lift", "can"])
def test_computing_features_does_not_change_the_rollout(task: str) -> None:
    spec, act_spec = load_specs(task)
    ext = FeatureExtractor(spec)
    trajs = []
    for compute in (True, False):
        env, xml, s0 = _start(task)
        reset_to_state(env, xml, s0)
        states = []
        for a in _actions(2, act_spec.dim):
            env.step(a)
            if compute:
                ext.from_sim(env.sim)
            states.append(np.array(env.sim.get_state().flatten()))
        trajs.append(np.array(states))
        env.close()
    np.testing.assert_array_equal(trajs[0], trajs[1])


@pytest.mark.parametrize("task", ["lift", "can"])
def test_current_robosuite_observations_equal_features(task: str) -> None:
    """With forward kinematics and a fresh cache, robosuite == features for every block."""
    spec, act_spec = load_specs(task)
    env, _, _ = _start(task)
    ext = FeatureExtractor(spec)
    stale_seen = 0.0
    for a in _actions(3, act_spec.dim):
        raw = env.step(a)[0]
        feats = ext.dict_from_sim(env.sim)
        stale_seen = max(stale_seen, float(np.abs(raw["robot0_eef_pos"] -
                                                  feats["robot0_eef_pos"]).max()))
        env.sim.forward()
        env._get_observations(force_update=True)
        fresh = env._get_observations(force_update=True)
        for f in spec.features:
            block = fresh[_robosuite_key(f)][f.dataset_ref.lo:f.dataset_ref.hi]
            np.testing.assert_array_equal(feats[f.name], block, err_msg=f.name)
    env.close()
    assert stale_seen > 0.0  # raw env.step observations lag the state by one physics substep
