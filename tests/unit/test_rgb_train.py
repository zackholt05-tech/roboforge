"""RGB study trainer (Session 8, D-078): shared encoder init, frame alignment, resume, configs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from roboforge.train.rgb import (
    RGBConfig,
    RGBTrainer,
    build_policy,
    config_path,
    load_rgb_windows,
    state_digest,
)

SMALL = {"observation.image_size": 16, "observation.crop": 12, "encoder.trunk": "compact",
         "train.batch_size": 4, "train.ckpt_every": 3, "train.log_every": 1,
         "train.max_steps": 6, "device.kind": "cpu"}


def _cfg(family: str, seed: int = 0, **extra: object) -> RGBConfig:
    return RGBConfig.load(config_path(family), {**SMALL, "seed": seed, **extra})


@pytest.fixture(scope="module")
def coded_frames(lift_processed_path: Path) -> dict:
    """Frame t of demo_k at 16²: channel 0 = t mod 256, channel 1 = k mod 256."""
    from roboforge.data.pipeline import load_task_artifacts
    from roboforge.data.processed import load_processed, processed_path

    proc = load_task_artifacts("lift")["task_manifest"]["processed"]
    eps = load_processed(processed_path("lift", proc["converter_version"]))
    assert processed_path("lift", proc["converter_version"]) == lift_processed_path
    out = {}
    for d, arr in eps.items():
        n, k = len(arr["obs"]), int(d.split("_")[1])
        f = np.zeros((n, 16, 16, 3), dtype=np.uint8)
        f[..., 0] = (np.arange(n) % 256)[:, None, None]
        f[..., 1] = k % 256
        out[d] = {"agentview": f, "robot0_eye_in_hand": f.copy()}
    return {"frames": out}


def test_both_families_share_the_encoder_initialization() -> None:
    a = build_policy(RGBConfig.load(config_path("bc_chunk"), {"seed": 1}))
    b = build_policy(RGBConfig.load(config_path("diffusion"), {"seed": 1}))
    c = build_policy(RGBConfig.load(config_path("bc_chunk"), {"seed": 2}))
    assert state_digest(a.enc) == state_digest(b.enc) != state_digest(c.enc)


def test_configs_differ_only_in_the_head_and_copy_level1_model_blocks() -> None:
    a = yaml.safe_load(config_path("bc_chunk").read_text())
    b = yaml.safe_load(config_path("diffusion").read_text())
    head_keys = {"name", "family", "head_family", "head_config", "model", "loss"}
    assert {k for k in a if a[k] != b.get(k)} == head_keys
    for cfg in (a, b):
        l1 = yaml.safe_load((config_path("bc_chunk").parents[3] / cfg["head_config"]).read_text())
        assert cfg["model"] == l1["model"] and cfg["loss"] == l1["loss"]
        assert cfg["optim"] == {**l1["optim"]}  # Level 1 default optimizer, untuned
        assert cfg["horizons"] == l1["horizons"]


def test_windows_read_the_aligned_frames(coded_frames: dict) -> None:
    cfg = _cfg("bc_chunk")
    w = load_rgb_windows(cfg, "train", coded_frames)
    from roboforge.data.sequence import SequenceDataset
    from roboforge.train.data import WindowTensors

    lw = WindowTensors.from_dataset(SequenceDataset.from_committed("lift", "train",
                                                                   obs_horizon=2,
                                                                   pred_horizon=16))
    idx = torch.tensor([0, 1, 2, 57, 500, len(w) - 1])
    b = w.batch(idx, ("agentview", "robot0_eye_in_hand"))
    assert b["images"].shape == (6, 2, 2, 16, 16, 3)
    for j, i in enumerate(idx.tolist()):
        t, d = int(lw.start_t[i]), lw.episode_id[i]
        want_t = [max(0, t - 1), t]
        assert b["images"][j, :, 0, 0, 0, 0].tolist() == [x % 256 for x in want_t]
        assert int(b["images"][j, 0, 1, 0, 0, 1]) == int(d.split("_")[1]) % 256
        assert b["obs_valid"][j].tolist() == [t >= 1, True]
    assert torch.equal(w.proprio, lw.obs_history[..., :9])
    assert torch.equal(w.future_action, lw.future_action)


def _trainer(tmp: Path, family: str, frames: dict) -> RGBTrainer:
    cfg = _cfg(family)
    ctx = {"run_id": "test", "normalization": {}, "hashes": {}}
    return RGBTrainer(cfg, tmp, load_rgb_windows(cfg, "train", frames),
                      load_rgb_windows(cfg, "valid", frames), ctx, "cpu")


@pytest.mark.parametrize("family", ["bc_chunk", "diffusion"])
def test_resume_equals_uninterrupted(tmp_path: Path, family: str, coded_frames: dict) -> None:
    full = _trainer(tmp_path / "full", family, coded_frames)
    assert full.run() == "complete" and full.step == 6
    cks = sorted(p.name for p in (tmp_path / "full" / "checkpoints").iterdir())
    assert cks == ["step_0000003.pt", "step_0000006.pt"]
    part = _trainer(tmp_path / "part", family, coded_frames)
    assert part.run(stop_at_step=4) == "interrupted"
    again = _trainer(tmp_path / "part", family, coded_frames)
    assert again.load_resume() == 4 and again.run() == "complete"
    assert state_digest(again.model) == state_digest(full.model)
    if family == "diffusion":
        assert state_digest(again.ema.averaged_model) == state_digest(full.ema.averaged_model)
    p = torch.load(tmp_path / "full" / "checkpoints" / "step_0000006.pt", weights_only=True)
    assert (p["ema_model"] is not None) == (family == "diffusion")
    full.save_cadence()  # a recomputed step keeps the earlier file aside
    assert (tmp_path / "full" / "checkpoints" / "step_0000006.orphan0.pt").is_file()
