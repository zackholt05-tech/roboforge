"""Observatory export (D-089): clips are recomputed without stepping; clone simulation is exact
and refuses every partition but validation.

All episodes here are recorded natively in the test from **validation** starts (records made on
another platform would not replay bitwise, KI-013), so no held-out state is ever touched.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from roboforge.data.pipeline import load_specs
from roboforge.eval.policies import RandomPolicy
from roboforge.eval.rollout import run_episode
from roboforge.sim.env_adapter import RoboForgeEnv
from roboforge.sim.eval_states import load_suite
from roboforge.viz import coords
from roboforge.viz.obs_branch import Brancher, check_partition, run_serial
from roboforge.viz.obs_scene import EEF_BODY, build_clip, scene_table

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def episode():
    init = load_suite("lift", "validation")[0]
    env = RoboForgeEnv("lift", horizon=12)
    rec = run_episode(env, RandomPolicy(load_specs("lift")[1], 3), init,
                      run_id="test_observatory")
    yield env, init, rec
    env.close()


def _request(init) -> dict:
    return {"id": "lift-random-s0-x", "kind": "test", "family": "random_uniform", "seed": 0,
            "state_id": init.state_id, "partition": init.partition, "roles": ["test"],
            "records_dir": "runs/test"}


def test_clip_never_steps_and_reproduces_recorded_poses(episode, monkeypatch) -> None:
    env, init, rec = episode
    env.reset(init)

    def no_step(*a, **k):
        raise AssertionError("the clip export stepped physics")

    monkeypatch.setattr(RoboForgeEnv, "step", no_step)
    man = {"eval_run_id": "test", "checkpoint": {}, "train_run_id": None}
    clip = build_clip(_request(init), rec.meta, rec.arrays, man, env.kinematics, env.model)
    v = clip["verification"]["recomputed_vs_recorded"]
    assert v["body_pos_max_abs_m"] == 0.0 and v["body_quat_max_abs"] == 0.0
    assert v["geom_pos_max_abs_m"] < 1e-12 and v["geom_rot_max_abs"] < 1e-12
    n, dyn = clip["frames"]["n"], clip["frames"]["dynamic_index"]
    assert n == len(rec.arrays["state"])
    pos = np.asarray(clip["frames"]["pos"])
    assert pos.shape == (n, len(dyn), 3)
    assert np.abs(pos - rec.arrays["body_pos"][:, dyn]).max() <= 0.5e-6  # 6-decimal rounding


def test_clip_refuses_missing_provenance(episode) -> None:
    env, init, rec = episode
    env.reset(init)
    meta = {**rec.meta, "physics_sha256": None}
    with pytest.raises(ValueError, match="provenance"):
        build_clip(_request(init), meta, rec.arrays, {"eval_run_id": "t", "checkpoint": {}},
                   env.kinematics, env.model)


def test_only_a_non_learned_policy_may_lack_a_checkpoint(episode) -> None:
    # KI-050: the random baseline's manifest has ``checkpoint: null``; that must be accepted
    # for a non-learned policy and refused for a learned one.
    env, init, rec = episode
    env.reset(init)
    man = {"eval_run_id": "t", "checkpoint": None}
    assert rec.meta["policy"]["learned"] is False
    clip = build_clip(_request(init), rec.meta, rec.arrays, man, env.kinematics, env.model)
    assert clip["checkpoint"]["sha256"] is None
    learned = {**rec.meta, "policy": {**rec.meta["policy"], "learned": True}}
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        build_clip(_request(init), learned, rec.arrays, man, env.kinematics, env.model)


def test_scene_geoms_compose_to_mujoco_world_poses(episode) -> None:
    import mujoco

    env, init, rec = episode
    env.reset(init)
    m, kin = env.model, env.kinematics
    geoms, meshes = scene_table(m, kin.body_names, kin.body_categories)
    assert any(g["role"] == "object" for g in geoms) and meshes
    d = mujoco.MjData(m)
    s = rec.arrays["state"][-1]
    d.qpos[:] = s[1:1 + m.nq]
    mujoco.mj_kinematics(m, d)
    f = kin.frame(s)
    bi = {n: i for i, n in enumerate(kin.body_names)}
    for g in geoms:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g["name"]) if g["name"] else -1
        if gid < 0:
            continue
        bp, bq = f["body_pos"][bi[g["body"]]], f["body_quat_wxyz"][bi[g["body"]]]
        wp = bp + coords.quat_to_mat(bq) @ np.asarray(g["pos"])
        assert np.abs(wp - d.geom_xpos[gid]).max() < 1e-12


def test_branch_of_executed_actions_reproduces_the_record(episode) -> None:
    _, init, rec = episode
    env = RoboForgeEnv("lift", horizon=12)
    try:
        b = Brancher(env, init, rec.arrays["executed_action"], rec.arrays["state"], EEF_BODY)
        f = 4
        got = run_serial(b, [(f, rec.arrays["executed_action"][f:f + 5], True)])[0]
        assert np.array_equal(got["states"], rec.arrays["state"][f + 1:f + 6])
        i = rec.meta["bodies"]["names"].index(EEF_BODY)
        assert np.array_equal(got["path"], rec.arrays["body_pos"][f:f + 6, i])
        other = run_serial(b, [(f, -rec.arrays["executed_action"][f:f + 5], False)])[0]
        assert other["states"] is None and not np.array_equal(other["path"], got["path"])
    finally:
        env.close()


def test_clone_simulation_refuses_held_out_and_robustness_partitions(episode) -> None:
    _, init, _ = episode
    check_partition(init)
    for part in ("eval_final_test", "eval_robustness", "demo_train"):
        with pytest.raises(PermissionError):
            check_partition(dataclasses.replace(init, partition=part))
