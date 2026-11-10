"""Level 2 point-cloud arm (Session 9): matched config, aligned windows, resume, trainer reuse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from roboforge.train import rgb as rgb_train
from roboforge.train.pointcloud import (
    PCConfig,
    PCTrainer,
    build_pc_policy,
    config_path,
    load_pc_windows,
)
from roboforge.train.rgb import state_digest

SMALL = {"train.batch_size": 4, "train.ckpt_every": 3, "train.log_every": 1,
         "train.max_steps": 6, "device.kind": "cpu"}


def _cfg(seed: int = 0) -> PCConfig:
    return PCConfig.load(config_path(), {**SMALL, "seed": seed})


@pytest.fixture(scope="module")
def coded_clouds(lift_processed_path: Path) -> dict:
    """Cloud t of demo_k: point 0 = (t, k, 0) / 1000 + the box centre; 8 points per cloud."""
    from roboforge.data.pipeline import load_task_artifacts
    from roboforge.data.processed import load_processed, processed_path

    spec = _cfg().cloud_spec
    proc = load_task_artifacts("lift")["task_manifest"]["processed"]
    eps = load_processed(processed_path("lift", proc["converter_version"]))
    assert processed_path("lift", proc["converter_version"]) == lift_processed_path
    out = {}
    for d, arr in eps.items():
        n, k = len(arr["obs"]), int(d.split("_")[1])
        p = np.tile(spec.centre.astype(np.float32), (n, 8, 1))
        p[:, 0, 0] += np.arange(n, dtype=np.float32) / 1000
        p[:, 0, 1] += k / 1000
        out[d] = p
    return {"points": out, "spec": spec.to_dict()}


def test_config_matches_the_rgb_diffusion_arm_except_the_observation() -> None:
    """Everything the freeze must match is identical to the frozen RGB diffusion config."""
    pc = yaml.safe_load(config_path().read_text())
    rgb = yaml.safe_load(rgb_train.config_path("diffusion").read_text())
    for k in ("head_family", "head_config", "task", "seed", "horizons", "model", "loss",
              "optim", "train", "device"):
        assert pc[k] == rgb[k], k
    for k in ("proprio", "proprio_dims"):
        assert pc["observation"][k] == rgb["observation"][k], k
    assert pc["observation"]["cloud"]["cameras"] == rgb["observation"]["cameras"]
    assert pc["observation"]["cloud"]["size"] == rgb["observation"]["image_size"]
    differing = {k for k in set(pc) | set(rgb) if pc.get(k) != rgb.get(k)}
    assert differing == {"schema", "name", "family", "observation", "encoder", "data"}


def test_windows_read_the_aligned_clouds(coded_clouds: dict) -> None:
    from roboforge.data.sequence import SequenceDataset
    from roboforge.train.data import WindowTensors

    cfg = _cfg()
    w = load_pc_windows(cfg, "train", coded_clouds)
    lw = WindowTensors.from_dataset(SequenceDataset.from_committed("lift", "train",
                                                                   obs_horizon=2,
                                                                   pred_horizon=16))
    idx = torch.tensor([0, 1, 2, 57, 500, len(w) - 1])
    b = w.batch(idx)
    assert b["points"].shape == (6, 2, 8, 3)
    c = cfg.cloud_spec.centre
    for j, i in enumerate(idx.tolist()):
        t, d = int(lw.start_t[i]), lw.episode_id[i]
        got_t = ((b["points"][j, :, 0, 0].double().numpy() - c[0]) * 1000).round().tolist()
        assert got_t == [max(0, t - 1), t]
        assert round(float(b["points"][j, 0, 0, 1] - c[1]) * 1000) == int(d.split("_")[1])
        assert b["obs_valid"][j].tolist() == [t >= 1, True]
    assert torch.equal(w.proprio, lw.obs_history[..., :9])
    assert torch.equal(w.future_action, lw.future_action)


def test_windows_refuse_clouds_built_with_another_spec(coded_clouds: dict) -> None:
    other = {**coded_clouds, "spec": {**coded_clouds["spec"], "n_points": 1024}}
    with pytest.raises(ValueError, match="another spec"):
        load_pc_windows(_cfg(), "train", other)


def _trainer(tmp: Path, clouds: dict) -> PCTrainer:
    cfg = _cfg()
    ctx = {"run_id": "test", "normalization": {}, "hashes": {}}
    return PCTrainer(cfg, tmp, load_pc_windows(cfg, "train", clouds),
                     load_pc_windows(cfg, "valid", clouds), ctx, "cpu")


def test_trainer_reuses_the_rgb_training_mechanics() -> None:
    for name in ("train_step", "save_cadence", "save_resume", "run"):
        assert getattr(PCTrainer, name) is getattr(rgb_train.RGBTrainer, name), name


def test_resume_equals_uninterrupted(tmp_path: Path, coded_clouds: dict) -> None:
    full = _trainer(tmp_path / "full", coded_clouds)
    assert full.run() == "complete" and full.step == 6
    cks = sorted(p.name for p in (tmp_path / "full" / "checkpoints").iterdir())
    assert cks == ["step_0000003.pt", "step_0000006.pt"]
    part = _trainer(tmp_path / "part", coded_clouds)
    assert part.run(stop_at_step=4) == "interrupted"
    again = _trainer(tmp_path / "part", coded_clouds)
    assert again.load_resume() == 4 and again.run() == "complete"
    assert state_digest(again.model) == state_digest(full.model)
    assert state_digest(again.ema.averaged_model) == state_digest(full.ema.averaged_model)
    p = torch.load(tmp_path / "full" / "checkpoints" / "step_0000006.pt", weights_only=True)
    assert p["schema"] == "roboforge.pc_checkpoint/1" and p["ema_model"] is not None


def test_same_seed_gives_the_rgb_arm_batch_and_noise_streams(coded_clouds: dict,
                                                            tmp_path: Path) -> None:
    """The sampler and diffusion-noise generator are seeded exactly as in the RGB arm."""
    from roboforge.train.data import EpochStreamSampler

    tr = _trainer(tmp_path / "s", coded_clouds)
    ref = EpochStreamSampler(len(tr.train), 4, 0)
    assert [tr.sampler.next_batch().tolist() for _ in range(3)] == \
        [ref.next_batch().tolist() for _ in range(3)]
    g = torch.Generator().manual_seed(int(np.random.SeedSequence([0, 3]).generate_state(1)[0]))
    assert torch.equal(tr.noise_gen.get_state(), g.get_state())


def test_seeded_construction_is_reproducible() -> None:
    assert state_digest(build_pc_policy(_cfg(1))) == state_digest(build_pc_policy(_cfg(1)))
    assert state_digest(build_pc_policy(_cfg(1))) != state_digest(build_pc_policy(_cfg(2)))
