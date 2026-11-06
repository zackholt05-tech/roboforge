"""Diffusion policy in the closed loop: seeded sampling, receding horizon, reload, isolation.

Needs the simulator and the processed data. The tiny run trained here is a pipeline
fixture, not a result.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from roboforge.eval.learned import LearnedPolicy, model_from_payload, noise_seed
from roboforge.eval.rollout import run_episode
from roboforge.paths import REPO_ROOT
from roboforge.sim.env_adapter import RoboForgeEnv
from roboforge.sim.eval_states import load_suite
from roboforge.train.checkpoint import load_checkpoint
from roboforge.train.config import load_config
from roboforge.train.run import train_run

pytestmark = [pytest.mark.sim, pytest.mark.data]


@pytest.fixture(scope="module")
def env():
    e = RoboForgeEnv("lift", horizon=40)
    yield e
    e.close()


@pytest.fixture(scope="module")
def run(tmp_path_factory, lift_lowdim_path):
    cfg = load_config(REPO_ROOT / "configs" / "models" / "diffusion_unet.yaml",
                      ["train.max_steps=40", "train.ckpt_every=40", "train.val_every=40",
                       "train.log_every=20"])
    return train_run(cfg, tmp_path_factory.mktemp("runs"), "test")


def _policy(m, **kw) -> LearnedPolicy:
    c = m["checkpoints"][-1]
    return LearnedPolicy.from_checkpoint(c["path"], expected_sha256=c["sha256"], **kw)


def test_manifest_records_diffusion_provenance(run) -> None:
    assert run["status"] == "complete" and run["reload_check"]["passed"]
    assert run["reload_check"]["inference_network"] == "ema"
    assert run["param_count"] == 1_092_551
    assert run["diffusion"]["sampler"]["timesteps"] == [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    assert "diffusion_noise_generator" in run["seeds"]


def test_first_decision_executes_the_first_t_a_actions_of_a_seeded_sample(env, run) -> None:
    policy = _policy(run)
    init = load_suite("lift", "validation")[0]
    rec = run_episode(env, policy, init, codec=policy.codec(env.action_spec))
    seed = noise_seed(0, init.state_id, 0)
    assert rec.meta["policy_episode"]["noise_seed"] == seed
    assert rec.meta["policy"]["params"]["policy_seed"] == 0
    # Recompute decision 0 independently from frame 0 (window: frame 0 repeated, padded).
    x = torch.from_numpy(np.stack([policy._normalize(rec.arrays["obs"][0])] * 2))[None]
    v = torch.tensor([[False, True]])
    chunk = policy.model.sample(x, v, torch.Generator().manual_seed(seed))[0].double()
    n = min(8, len(rec.arrays["raw_policy_action"]))
    np.testing.assert_array_equal(rec.arrays["raw_policy_action"][:n], chunk[:n].numpy())
    assert rec.arrays["decision_index"][:n].tolist() == [0] * n
    if len(rec.arrays["decision_index"]) > 8:
        assert rec.arrays["decision_index"][8] == 1  # replans after T_a = 8 actions


def test_reload_gives_identical_closed_loop_physics_and_seeds_isolate_episodes(env, run):
    c = run["checkpoints"][-1]
    payload = load_checkpoint(c["path"], expected_sha256=c["sha256"])
    in_memory = LearnedPolicy(model_from_payload(payload), payload,
                              checkpoint_sha256=c["sha256"])
    states = load_suite("lift", "validation")
    codec = in_memory.codec(env.action_spec)
    r1 = run_episode(env, in_memory, states[1], codec=codec)
    reloaded = _policy(run)
    run_episode(env, reloaded, states[2], codec=codec)  # a different episode first
    r2 = run_episode(env, reloaded, states[1], codec=codec)
    assert r1.meta["physics_sha256"] == r2.meta["physics_sha256"]
    np.testing.assert_array_equal(r1.arrays["raw_policy_action"],
                                  r2.arrays["raw_policy_action"])
    other = run_episode(env, _policy(run, repeat=1), states[1], codec=codec)
    assert other.meta["policy_episode"]["noise_seed"] != r1.meta["policy_episode"]["noise_seed"]
    assert not np.array_equal(other.arrays["raw_policy_action"][:8],
                              r1.arrays["raw_policy_action"][:8])
    # Every decoded action lies in the action bounds without clipping the motion dims:
    # sampled chunks are clipped to the normalized range [-1, 1] by the sampler.
    assert not r1.arrays["clip_mask"][:, :6].any()


def test_denoising_export_replays_the_recorded_sampler_calls_bitwise(env, run) -> None:
    from roboforge.viz.denoising import decision_frames, decision_trace, replay_episode

    policy = _policy(run)
    codec = policy.codec(env.action_spec)
    init = load_suite("lift", "validation")[4]
    rec = run_episode(env, policy, init, codec=codec)
    frames = decision_frames(rec)
    caps = replay_episode(_policy(run), rec, {0, len(frames) - 1})
    meta, arrays = decision_trace(policy, rec, caps[0], codec)
    assert all(meta["verification"].values()), meta["verification"]
    assert arrays["timesteps"].tolist() == [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    assert arrays["sample_in_normalized"].shape == (10, 16, 7)
    n = meta["n_executed"]
    np.testing.assert_array_equal(arrays["final_chunk_normalized"][:n],
                                  rec.arrays["raw_policy_action"][:n])
    np.testing.assert_array_equal(arrays["final_chunk_env_action"][:n],
                                  rec.arrays["executed_action"][:n])
    assert meta["noise"]["episode_noise_seed"] == meta["noise"]["seed_check"]
    assert "proposal" in meta["labels"]["intermediate"]
    tampered = type(rec)(rec.meta, {**rec.arrays,
                                    "raw_policy_action": rec.arrays["raw_policy_action"] + 1e-9})
    with pytest.raises(AssertionError):
        replay_episode(_policy(run), tampered, {0})
