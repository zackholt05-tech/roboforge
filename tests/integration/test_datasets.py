"""Integration tests on the real robomimic PH datasets (skipped when not downloaded)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from roboforge.data.fetch import load_manifest, verify_file
from roboforge.data.inspect import env_args, summarize_hdf5
from roboforge.sim.factory import load_sim_config, make_env, task_kwargs
from roboforge.sim.state_replay import replay_demo, reset_to_state

pytestmark = pytest.mark.data


@pytest.mark.parametrize("task", ["lift", "can"])
def test_files_match_manifest_hash(task: str) -> None:
    _, files = load_manifest()
    for entry in (f for f in files if f.task == task):
        path = entry.destination()
        if not path.exists():
            pytest.skip(f"{path} not downloaded")
        assert verify_file(entry, path) == entry.sha256


@pytest.mark.parametrize("task", ["lift", "can"])
def test_sim_config_equals_dataset_metadata(task: str, request) -> None:
    path = request.getfixturevalue(f"{task}_lowdim_path")
    with h5py.File(path, "r") as f:
        meta = env_args(f)
    cfg = load_sim_config(task)
    assert cfg["env_name"] == meta["env_name"]
    assert cfg["env_version"] == meta["env_version"]
    assert task_kwargs(cfg) == task_kwargs(meta)


def test_lift_structure(lift_lowdim_path) -> None:
    s = summarize_hdf5(lift_lowdim_path)
    assert s["n_demos"] == 200 and s["action_dim"] == 7 and s["state_dim"] == 32
    assert s["action_abs_le_1"] and s["num_samples_matches_action_rows"]
    assert s["gripper_action_unique_values"] == [-1.0, 1.0]
    assert s["mask_sizes"]["train"] == 180 and s["mask_sizes"]["valid"] == 20
    assert set(s["masks"]["train"]).isdisjoint(s["masks"]["valid"])
    assert s["obs_keys"]["object"] == [10]


def test_lift_obs_recomputed_from_state(lift_lowdim_path) -> None:
    env = make_env(load_sim_config("lift"))
    with h5py.File(lift_lowdim_path, "r") as f:
        g = f["data"]["demo_3"]
        for t in (0, 20, len(g["actions"]) - 1):
            reset_to_state(env, g.attrs["model_file"], g["states"][t])
            obs = env._get_observations(force_update=True)
            np.testing.assert_allclose(
                obs["robot0_eef_pos"], g["obs"]["robot0_eef_pos"][t], atol=1e-9
            )
            np.testing.assert_allclose(obs["object-state"], g["obs"]["object"][t], atol=1e-9)
    env.close()


def test_lift_open_loop_replay_demo0(lift_lowdim_path) -> None:
    env = make_env(load_sim_config("lift"))
    with h5py.File(lift_lowdim_path, "r") as f:
        g = f["data"]["demo_0"]
        r = replay_demo(env, "demo_0", g.attrs["model_file"], g["states"][()], g["actions"][()],
                        task_objects=["cube_joint0"])
    env.close()
    assert r.reset_state_max_abs == 0.0
    assert r.dataset_final_state_success and r.replay_success_at_end
    assert r.max_errors["object_pos_max_abs_m"] < 0.01  # 1 cm tolerance (Session 0 observed ~1 mm)
