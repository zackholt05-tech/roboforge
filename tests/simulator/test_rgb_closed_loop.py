"""RGB-study closed-loop policy (Session 8): tiny trained checkpoints drive real episodes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from roboforge.eval.rgb import RGBClosedLoopPolicy, load_rgb_checkpoint
from roboforge.eval.rollout import run_episode
from roboforge.sim.env_adapter import RoboForgeEnv
from roboforge.sim.eval_states import load_suite
from roboforge.train.rgb import RGBConfig, RGBTrainer, config_path, load_rgb_windows
from roboforge.train.run import data_context

pytestmark = [pytest.mark.sim, pytest.mark.data, pytest.mark.render]

SMALL = {"observation.image_size": 16, "observation.crop": 12, "encoder.trunk": "compact",
         "train.batch_size": 4, "train.ckpt_every": 2, "train.log_every": 1,
         "train.max_steps": 2, "device.kind": "cpu"}


@pytest.fixture(scope="module")
def frames() -> dict:
    from roboforge.data.pipeline import load_task_artifacts
    from roboforge.data.processed import load_processed, processed_path

    proc = load_task_artifacts("lift")["task_manifest"]["processed"]
    eps = load_processed(processed_path("lift", proc["converter_version"]))
    rng = np.random.default_rng(0)
    return {"frames": {d: {c: rng.integers(0, 256, (len(a["obs"]), 16, 16, 3), dtype=np.uint8)
                           for c in ("agentview", "robot0_eye_in_hand")}
                       for d, a in eps.items()}}


@pytest.fixture(scope="module")
def env():
    e = RoboForgeEnv("lift", horizon=40)
    yield e
    e.close()


@pytest.mark.parametrize("family", ["bc_chunk", "diffusion"])
def test_policy_runs_an_episode_rendering_only_decision_frames(
        tmp_path: Path, family: str, frames: dict, env: RoboForgeEnv) -> None:
    cfg = RGBConfig.load(config_path(family), SMALL)
    ctx = data_context("lift", "test")
    tr = RGBTrainer(cfg, tmp_path, load_rgb_windows(cfg, "train", frames),
                    load_rgb_windows(cfg, "valid", frames), ctx, "cpu")
    ck = {}
    tr.run(on_checkpoint=lambda c: ck.update(c))
    payload = load_rgb_checkpoint(tmp_path / ck["path"], ck["sha256"])
    pol = RGBClosedLoopPolicy(payload, ck["sha256"])
    pol.bind(env)
    init = load_suite("lift", "validation")[0]
    rec = run_episode(env, pol, init, run_id="test", codec=pol.codec(env.action_spec))
    assert rec.outcome["stop_reason"] != "error", rec.outcome["error"]
    assert rec.outcome["error"] is None
    assert int(rec.arrays["executed_action"].shape[0]) == 40
    # decisions at frames 0, 8, ..., 32: frame 0 alone, then frames 7-8, ..., 31-32, and
    # 39-40 for the decision the 40-step horizon pre-empts; never any other frame
    assert pol.n_decisions == 5 and pol.n_renders == 1 + 2 * 5
    assert pol.info.params["inference_network"] == ("ema" if family == "diffusion"
                                                    else "trained")
    if family == "diffusion":
        assert rec.meta["policy_episode"]["noise_seed"] is not None
    assert torch.get_num_threads() == 1
