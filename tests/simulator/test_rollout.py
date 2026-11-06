"""Closed-loop rollout recorder: frame indexing, outcomes, errors, re-simulation audit."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from roboforge.eval.policies import DemoReplayPolicy, PolicyInfo, RandomPolicy
from roboforge.eval.rollout import RenderSpec, RolloutRecord, run_episode, verify_record
from roboforge.sim.env_adapter import RoboForgeEnv
from roboforge.sim.eval_states import load_suite
from roboforge.sim.snapshot import from_demo

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def lift():
    env = RoboForgeEnv("lift", horizon=30)
    yield env
    env.close()


class _Scripted:
    """Test policy: fixed chunk of k actions per decision, or a NaN at decision ``bad_at``."""

    def __init__(self, k: int, bad_at: int | None = None) -> None:
        self.k, self.bad_at, self.n = k, bad_at, 0
        self.info = PolicyInfo("scripted_test", "test", False, True, "env", "unit-test policy")

    def reset_episode(self, episode) -> None:
        self.n = 0

    def act(self, obs) -> np.ndarray:
        self.n += 1
        a = np.tile(np.linspace(-0.5, 0.5, 7), (self.k, 1)) * (1 if self.n % 2 else -1)
        if self.bad_at is not None and self.n == self.bad_at:
            a[0, 3] = np.nan
        return a


def test_random_rollout_is_recorded_frame_indexed_and_resimulates(lift, tmp_path) -> None:
    init = load_suite("lift", "validation")[0]
    rec = run_episode(lift, RandomPolicy(lift.action_spec, 0), init, run_id="t",
                      render=RenderSpec(("frontview",), 32, 32, every=10))
    a = rec.arrays
    T = rec.outcome["steps"]
    assert T == 30 and rec.outcome["truncated"] and not rec.outcome["terminated"]
    assert rec.outcome["stop_reason"] == "truncated_horizon" and not rec.outcome["success"]
    for k in ("state", "obs", "time_s", "joint_qpos", "body_pos", "success"):
        assert len(a[k]) == T + 1, k
    for k in ("executed_action", "raw_policy_action", "reward", "clip_mask", "decision_index"):
        assert len(a[k]) == T, k
    np.testing.assert_array_equal(a["state"][0], init.state)
    np.testing.assert_allclose(np.diff(a["time_s"]), lift.control_dt, atol=1e-12)
    assert a["rgb_frontview"].shape == (4, 32, 32, 3) and a["rgb_frame_index"].tolist() == [0, 10,
                                                                                         20, 30]
    assert rec.meta["policy"]["autonomous"] and not rec.meta["policy"]["learned"]
    assert rec.meta["episode"]["initial_state"]["partition"] == "eval_validation"
    assert verify_record(rec, init)["passed"]
    rec.save(tmp_path / "r")
    back = RolloutRecord.load(tmp_path / "r")
    assert back.meta["physics_sha256"] == rec.meta["physics_sha256"]


def test_action_chunks_are_indexed_by_decision(lift) -> None:
    rec = run_episode(lift, _Scripted(k=4), load_suite("lift", "validation")[1])
    a = rec.arrays
    assert a["decision_index"][:9].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2]
    assert a["chunk_offset"][:5].tolist() == [0, 1, 2, 3, 0]
    assert np.isnan(a["planning_latency_ms"][1]) and not np.isnan(a["planning_latency_ms"][4])


def test_invalid_policy_output_is_a_recorded_failure(lift) -> None:
    rec = run_episode(lift, _Scripted(k=1, bad_at=3), load_suite("lift", "validation")[2])
    out = rec.outcome
    assert out["stop_reason"] == "error" and out["error"]["type"] == "InvalidActionError"
    assert out["steps"] == 2 and not out["success"]
    assert len(rec.arrays["state"]) == 3


@pytest.mark.data
@pytest.mark.parametrize("hold", [0, 3])
def test_demo_replay_terminates_at_first_success(lift_lowdim_path, hold: int) -> None:
    with h5py.File(lift_lowdim_path, "r") as f:
        g = f["data"]["demo_0"]
        init = from_demo("lift", "demo_0", g.attrs["model_file"], g["states"][()],
                         "demo_valid", "-")
        actions = g["actions"][()]
    env = RoboForgeEnv("lift", hold_steps=hold)
    rec = run_episode(env, DemoReplayPolicy(actions, "demo_0", "-"), init)
    env.close()
    out = rec.outcome
    assert rec.meta["policy"]["autonomous"] is False
    assert out["success"] and out["terminated"] and out["stop_reason"] == "terminated_success"
    assert out["steps"] == out["first_success_step"]
    assert out["hold_steps_executed"] == hold
    assert rec.arrays["success"][out["first_success_step"]]
    assert not rec.arrays["success"][:out["first_success_step"]].any()
    assert int(rec.arrays["clip_mask"].sum()) == 0
    if hold:
        assert out["success_held_through_hold"] is True
