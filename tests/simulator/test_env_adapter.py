"""RoboForgeEnv contract tests (needs robosuite + MuJoCo + committed eval states; no dataset)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from roboforge.data.features import FeatureExtractor
from roboforge.data.pipeline import load_specs
from roboforge.eval.policies import NoOpPolicy, RandomPolicy
from roboforge.eval.rollout import run_episode
from roboforge.sim import eval_states as es
from roboforge.sim.env_adapter import EpisodeDoneError, InvalidActionError, RoboForgeEnv
from roboforge.sim.eval_states import FinalTestLockedError, load_suite
from roboforge.sim.replay_compare import bare_replay

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module", params=["lift", "can"])
def task_env(request):
    env = RoboForgeEnv(request.param)
    yield request.param, env
    env.close()


def test_action_dimension_comes_from_the_committed_spec(task_env) -> None:
    task, env = task_env
    _, act = load_specs(task)
    assert env.action_dim == act.dim == env.raw_env.action_dim
    env.reset(load_suite(task, "validation")[0])
    obs, reward, term, trunc, info = env.step(np.zeros(act.dim))
    assert obs.vector.shape == (env.observation_spec.dim,) and obs.step_index == 1
    assert reward == 0.0 and not term and not trunc


@pytest.mark.parametrize("bad", [np.zeros(6), np.zeros(8), np.zeros((1, 7)),
                                 np.array([0, 0, 0, np.nan, 0, 0, 0]),
                                 np.array([np.inf, 0, 0, 0, 0, 0, 0])])
def test_invalid_actions_are_rejected_not_repaired(task_env, bad: np.ndarray) -> None:
    task, env = task_env
    env.reset(load_suite(task, "validation")[1])
    before = env.current_state()
    with pytest.raises(InvalidActionError):
        env.step(bad)
    np.testing.assert_array_equal(env.current_state(), before)


def test_out_of_bounds_actions_are_clipped_and_reported(task_env) -> None:
    task, env = task_env
    env.reset(load_suite(task, "validation")[2])
    a = np.array([2.0, -3.0, 0.5, 0.0, 1.5, -1.0, 7.0])
    _, _, _, _, info = env.step(a)
    np.testing.assert_array_equal(info["executed_action"], np.clip(a, -1, 1))
    assert info["clip_mask"].tolist() == [True, True, False, False, True, False, True]


def test_reset_restores_the_snapshot_exactly(task_env) -> None:
    task, env = task_env
    init = load_suite(task, "validation")[3]
    obs, meta = env.reset(init)
    np.testing.assert_array_equal(env.current_state(), init.state)
    ref = FeatureExtractor(env.observation_spec).bind(env.model).from_flat_state(init.state)
    np.testing.assert_array_equal(obs.vector, ref)
    assert meta.initial_state["partition"] == "eval_validation" and not meta.initial_success
    assert env.serialize_initial_state()["state_sha256"] == init.state_sha256


def test_seed_resets_and_the_final_test_seed_lock(task_env, monkeypatch) -> None:
    task, env = task_env
    _, meta = env.reset(100003)
    stored = load_suite(task, "validation")[3]
    assert meta.initial_state["model_xml_sha256"] == stored.model_xml_sha256
    assert meta.initial_state["partition"] == "eval_validation"
    # The lock points at a freeze that does not exist, so the refusal is tested whatever the real
    # lock state is: once docs/FINAL_TEST_FREEZE.md is committed the real lock is open, and no
    # test may build a held-out start (only scripts/eval_heldout.py reads them, D-067).
    cfg = copy.deepcopy(es.suites_config())
    cfg["suites"][es.LOCKED_SUITE]["locked_until"] = "docs/NO_SUCH_FREEZE.md"
    monkeypatch.setattr(es, "suites_config", lambda: cfg)
    unlocked, why = es.lock_status(task)
    assert not unlocked and "does not exist" in why
    with pytest.raises(FinalTestLockedError):
        env.reset(900003)


def test_truncation_at_the_horizon_and_no_steps_after() -> None:
    env = RoboForgeEnv("lift", horizon=5)
    env.reset(load_suite("lift", "validation")[0])
    flags = [env.step(np.zeros(7))[2:4] for _ in range(5)]
    assert flags == [(False, False)] * 4 + [(False, True)]
    with pytest.raises(EpisodeDoneError):
        env.step(np.zeros(7))
    env.close()


def test_recording_and_rendering_do_not_change_the_physics(task_env) -> None:
    task, env = task_env
    init = load_suite(task, "validation")[4]
    env.horizon = 60
    try:
        rec = run_episode(env, RandomPolicy(env.action_spec, 1), init)
        env.render("agentview", 48, 48)
        env.get_success()
    finally:
        env.horizon = env.task_spec["eval_horizon_steps"]
    bare = bare_replay(env.raw_env, init.model_xml, init.state[None], rec.arrays["executed_action"])
    np.testing.assert_array_equal(bare, rec.arrays["state"])


def test_episodes_are_isolated_within_one_instance(task_env) -> None:
    task, env = task_env
    val = load_suite(task, "validation")
    env.horizon = 40
    try:
        first = run_episode(env, RandomPolicy(env.action_spec, 2), val[5])
        run_episode(env, NoOpPolicy(env.action_spec), val[6])
        again = run_episode(env, RandomPolicy(env.action_spec, 2), val[5])
    finally:
        env.horizon = env.task_spec["eval_horizon_steps"]
    for k in ("state", "obs", "executed_action", "body_pos"):
        np.testing.assert_array_equal(first.arrays[k], again.arrays[k], err_msg=k)


def test_kinematics_and_constants(task_env) -> None:
    task, env = task_env
    env.reset(load_suite(task, "validation")[0])
    robot = env.get_robot_kinematics()
    assert robot["joint_names"][:7] == [f"robot0_joint{i}" for i in range(1, 8)]
    assert {f"robot0_link{i}" for i in range(1, 8)} <= set(robot["link_names"])
    assert robot["link_quat_wxyz"].shape == (len(robot["link_names"]), 4)
    scene = env.get_scene_kinematics()
    obj = {"lift": "cube_main", "can": "Can_main"}[task]
    assert obj in scene["body_names"]
    assert not {"left_eef_target", "right_eef_target"} & set(robot["link_names"]
                                                             + scene["body_names"])
    c = env.constants()
    assert c["control_freq_hz"] == 20.0 and c["substeps_per_control"] == 25
    assert c["horizon"] == 400 and c["termination"] == "first_success"
    assert c["action_spec"]["dim"] == 7 and c["privileged_features"]


@pytest.mark.render
def test_shadow_render_matches_robosuite_render_on_lift() -> None:
    from roboforge.sim.factory import load_sim_config, make_env
    from roboforge.sim.renderer import render_rgb
    from roboforge.sim.snapshot import restore

    init = load_suite("lift", "validation")[0]
    env = RoboForgeEnv("lift")
    env.reset(init)
    ours = env.render("frontview", 96, 96)
    rs = make_env(load_sim_config("lift"), offscreen=True)
    restore(rs, init)
    theirs = render_rgb(rs, "frontview", 96, 96)
    assert ours.dtype == np.uint8 and ours.shape == (96, 96, 3)
    np.testing.assert_array_equal(ours, theirs)
    rs.close()
    env.close()
