"""LearnedPolicy in the closed loop: train/inference parity, isolation, reload equivalence.

Needs the simulator and the processed data. The tiny models trained here are pipeline
fixtures, not results.
"""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from roboforge.data.pipeline import load_task_artifacts
from roboforge.data.processed import load_processed, processed_path
from roboforge.data.sequence import SequenceDataset
from roboforge.eval.learned import LearnedPolicy
from roboforge.eval.rollout import run_episode
from roboforge.paths import REPO_ROOT
from roboforge.sim.env_adapter import RoboForgeEnv
from roboforge.sim.eval_states import load_suite
from roboforge.sim.snapshot import from_demo
from roboforge.train.checkpoint import CheckpointError
from roboforge.train.config import load_config
from roboforge.train.run import train_run

pytestmark = [pytest.mark.sim, pytest.mark.data]
CONFIGS = REPO_ROOT / "configs" / "models"


@pytest.fixture(scope="module")
def processed(lift_lowdim_path):
    m = load_task_artifacts("lift")["task_manifest"]
    path = processed_path("lift", m["processed"]["converter_version"])
    if not path.exists():
        pytest.skip("processed data not built (run `make build-data`)")
    return load_processed(path, expected_sha256=m["processed"]["content_sha256"])


@pytest.fixture(scope="module")
def env():
    e = RoboForgeEnv("lift", horizon=40)
    yield e
    e.close()


@pytest.fixture(scope="module")
def runs(tmp_path_factory, processed):
    """Tiny trained runs (40 steps) of each primary configuration."""
    out = tmp_path_factory.mktemp("runs")
    made = {}
    for name in ("bc_mlp", "bc_rnn", "bc_mlp_chunk"):
        cfg = load_config(CONFIGS / f"{name}.yaml",
                          ["train.max_steps=40", "train.ckpt_every=40", "train.val_every=40",
                           "train.log_every=20"])
        made[name] = train_run(cfg, out, "test")
    return made


def _policy(manifest) -> LearnedPolicy:
    c = manifest["checkpoints"][-1]
    return LearnedPolicy.from_checkpoint(c["path"], expected_sha256=c["sha256"])


def test_simulator_observation_equals_processed_training_observation(env, processed,
                                                                     lift_lowdim_path) -> None:
    """Input parity: the env's t = 0 features, cast to float32, are the stored training obs."""
    with h5py.File(lift_lowdim_path, "r") as f:
        for demo in ("demo_0", "demo_5"):  # validation demos
            g = f["data"][demo]
            init = from_demo("lift", demo, g.attrs["model_file"], g["states"][()],
                             "demo_valid", "test")
            obs, _ = env.reset(init)
            np.testing.assert_array_equal(obs.vector.astype(np.float32),
                                          processed[demo]["obs"][0])


@pytest.mark.parametrize("name", ["bc_mlp", "bc_rnn", "bc_mlp_chunk"])
def test_policy_windows_equal_training_windows(runs, processed, name) -> None:
    """Window parity: feeding a demo's frames gives exactly SequenceDataset's windows."""
    policy = _policy(runs[name])
    t_o = policy.obs_horizon
    ds = SequenceDataset.from_committed("lift", "valid", obs_horizon=t_o, pred_horizon=1)
    demo = ds.demos[0]
    policy.reset_episode(SimpleNamespace(task="lift", **{k: policy.hashes[k] for k in (
        "observation_spec_sha256", "action_spec_sha256", "task_spec_sha256")}))
    for t, vec in enumerate(processed[demo]["obs"]):
        policy.observe(SimpleNamespace(vector=vec.astype(np.float64)))
        oh, ov = policy.window()
        item = ds[t]  # demo is the first valid demo, so item t is its step t
        assert item["episode_id"] == demo and item["start_t"] == t
        np.testing.assert_array_equal(oh, item["obs_history"])
        np.testing.assert_array_equal(ov, item["obs_valid"])


def test_policy_history_is_every_recorded_frame_and_chunks_execute_t_a(env, runs) -> None:
    policy = _policy(runs["bc_mlp_chunk"])
    init = load_suite("lift", "validation")[0]
    rec = run_episode(env, policy, init, codec=policy.codec(env.action_spec))
    a = rec.arrays
    expected = np.stack([policy._normalize(v) for v in a["obs"]])
    np.testing.assert_array_equal(np.stack(policy._hist), expected)
    assert rec.meta["policy"]["learned"] and rec.meta["policy"]["action_space"] == "normalized"
    assert rec.meta["policy"]["params"]["checkpoint_sha256"] == runs["bc_mlp_chunk"][
        "checkpoints"][-1]["sha256"]
    n = len(a["decision_index"])
    assert a["chunk_offset"].tolist() == [i % 8 for i in range(n)]
    np.testing.assert_array_equal(a["executed_action"],
                                  policy.codec(env.action_spec).decode(
                                      a["raw_policy_action"])[0])


@pytest.mark.parametrize("name", ["bc_mlp", "bc_rnn"])
def test_reloaded_checkpoint_gives_identical_closed_loop_physics(env, runs, name) -> None:
    """In-memory trained model vs the checkpoint reloaded strictly: same physics_sha256."""
    from roboforge.eval.learned import model_from_payload
    from roboforge.train.checkpoint import load_checkpoint

    c = runs[name]["checkpoints"][-1]
    payload = load_checkpoint(c["path"], expected_sha256=c["sha256"])
    in_memory = LearnedPolicy(model_from_payload(payload), payload,
                              checkpoint_sha256=c["sha256"])
    reloaded = _policy(runs[name])
    init = load_suite("lift", "validation")[1]
    r1 = run_episode(env, in_memory, init, codec=in_memory.codec(env.action_spec))
    r2 = run_episode(env, reloaded, init, codec=reloaded.codec(env.action_spec))
    assert r1.meta["physics_sha256"] == r2.meta["physics_sha256"]
    np.testing.assert_array_equal(r1.arrays["raw_policy_action"],
                                  r2.arrays["raw_policy_action"])


def test_no_state_carries_between_episodes(env, runs) -> None:
    policy = _policy(runs["bc_rnn"])
    states = load_suite("lift", "validation")
    codec = policy.codec(env.action_spec)
    fresh = run_episode(env, _policy(runs["bc_rnn"]), states[3], codec=codec)
    run_episode(env, policy, states[2], codec=codec)  # episode A first
    after = run_episode(env, policy, states[3], codec=codec)
    assert fresh.meta["physics_sha256"] == after.meta["physics_sha256"]


def test_policy_refuses_another_task_and_act_without_observe(runs) -> None:
    policy = _policy(runs["bc_mlp"])
    with pytest.raises(ValueError):
        policy.reset_episode(SimpleNamespace(task="can"))
    policy._hist = []
    with pytest.raises(RuntimeError):
        policy.act(SimpleNamespace(vector=np.zeros(19)))
    with pytest.raises(CheckpointError):
        LearnedPolicy.from_checkpoint(runs["bc_mlp"]["checkpoints"][-1]["path"],
                                      expected_sha256="0" * 64)
