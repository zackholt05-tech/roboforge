"""Level 2 closed-loop policy (Session 9): a tiny trained point-cloud checkpoint drives a real
episode, and its clouds are the dataset builder's clouds for the same state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from roboforge.data.pointcloud import CloudMaker
from roboforge.data.rgb import DemoFrameRenderer, iter_raw_demos, raw_file
from roboforge.eval.pointcloud import PCClosedLoopPolicy, load_pc_checkpoint
from roboforge.eval.rollout import run_episode
from roboforge.sim.env_adapter import RoboForgeEnv
from roboforge.sim.eval_states import load_suite
from roboforge.sim.snapshot import InitialState
from roboforge.train.pointcloud import PCConfig, PCTrainer, config_path, load_pc_windows
from roboforge.train.run import data_context

pytestmark = [pytest.mark.sim, pytest.mark.data, pytest.mark.render]

SMALL = {"train.batch_size": 4, "train.ckpt_every": 2, "train.log_every": 1,
         "train.max_steps": 2, "device.kind": "cpu"}


@pytest.fixture(scope="module")
def clouds() -> dict:
    from roboforge.data.pipeline import load_task_artifacts
    from roboforge.data.processed import load_processed, processed_path

    spec = PCConfig.load(config_path()).cloud_spec
    proc = load_task_artifacts("lift")["task_manifest"]["processed"]
    eps = load_processed(processed_path("lift", proc["converter_version"]))
    rng = np.random.default_rng(0)
    lo, hi = np.asarray(spec.box_lo), np.asarray(spec.box_hi)
    return {"points": {d: (lo + rng.random((len(a["obs"]), 16, 3)) * (hi - lo)).astype(np.float32)
                       for d, a in eps.items()}, "spec": spec.to_dict()}


@pytest.fixture(scope="module")
def env():
    e = RoboForgeEnv("lift", horizon=40)
    yield e
    e.close()


def test_policy_runs_an_episode_building_clouds_only_for_decision_frames(
        tmp_path: Path, clouds: dict, env: RoboForgeEnv) -> None:
    cfg = PCConfig.load(config_path(), SMALL)
    ctx = data_context("lift", "test")
    tr = PCTrainer(cfg, tmp_path, load_pc_windows(cfg, "train", clouds),
                   load_pc_windows(cfg, "valid", clouds), ctx, "cpu")
    ck = {}
    tr.run(on_checkpoint=lambda c: ck.update(c))
    payload = load_pc_checkpoint(tmp_path / ck["path"], ck["sha256"])
    pol = PCClosedLoopPolicy(payload, ck["sha256"])
    pol.bind(env)
    init = load_suite("lift", "validation")[0]
    try:
        rec = run_episode(env, pol, init, run_id="test", codec=pol.codec(env.action_spec))
    finally:
        pol.close()
    assert rec.outcome["stop_reason"] != "error", rec.outcome["error"]
    assert int(rec.arrays["executed_action"].shape[0]) == 40
    # decisions at frames 0, 8, ..., 32 (+ the one the horizon pre-empts): frame 0 alone, then
    # frames d + 7 and d + 8: the RGB arm's rule
    assert pol.n_decisions == 5 and pol.n_renders == 1 + 2 * 5
    assert pol.info.params["inference_network"] == "ema"
    assert rec.meta["policy_episode"]["noise_seed"] is not None
    assert torch.get_num_threads() == 1


def test_closed_loop_cloud_equals_the_dataset_builders_cloud() -> None:
    """The same state through the env's model (closed loop) and the demo renderer (dataset)."""
    try:
        raw_file("lift", "raw")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    spec = PCConfig.load(config_path()).cloud_spec
    demo = next(iter_raw_demos("lift", ["demo_0"]))
    r = DemoFrameRenderer("lift")
    try:
        r.load_model(demo.model_xml, demo.states[0])
        mk = CloudMaker(r.model, spec)
        want = [mk.cloud(demo.states[t])[0] for t in (0, 30)]
        mk.close()
    finally:
        r.close()
    e = RoboForgeEnv("lift")
    try:
        got = []
        for t in (0, 30):
            init = InitialState(task="lift", state_id=f"demo_0_t{t}", partition="demo_train",
                                model_xml=demo.model_xml, state=demo.states[t])
            e.reset(init)
            mk_e = CloudMaker(e.model, spec)
            got.append(mk_e.cloud(e.current_state())[0])
            mk_e.close()
    finally:
        e.close()
    for a, b in zip(want, got, strict=True):
        assert np.array_equal(a, b)
