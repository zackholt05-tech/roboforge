"""Trainer, sampler, config and checkpoints on synthetic windows (no dataset needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from roboforge.data.normalization import Normalizer
from roboforge.data.sequence import SequenceDataset
from roboforge.train.checkpoint import (
    CheckpointError,
    checkpoint_name,
    load_checkpoint,
    save_checkpoint,
)
from roboforge.train.config import apply_overrides, from_dict, run_id
from roboforge.train.data import EpochStreamSampler, WindowTensors
from roboforge.train.loop import Trainer, masked_mse

MODELS = {"bc_mlp": {"hidden": [64, 64], "activation": "relu", "layer_norm": False,
                     "dropout": 0.0},
          "bc_rnn": {"cell": "lstm", "hidden": 32, "layers": 2, "dropout": 0.0}}


def cfg_dict(family: str = "bc_mlp", t_o: int = 1, t_p: int = 1, **train) -> dict:
    t = {"max_steps": 40, "batch_size": 16, "ckpt_every": 20, "val_every": 20,
         "log_every": 10, "num_threads": 1, "grad_clip": None, **train}
    return {"schema": "roboforge.train_config/1", "name": f"t_{family}", "family": family,
            "task": "lift", "seed": 0, "horizons": {"obs": t_o, "pred": t_p, "exec": 1},
            "model": dict(MODELS[family]), "loss": {"kind": "mse"},
            "optim": {"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.0,
                      "schedule": "constant"}, "train": t}


def windows(t_o: int, t_p: int, lengths=(12, 30, 7)) -> WindowTensors:
    rng = np.random.default_rng(0)
    eps = {f"demo_{i}": {"obs": rng.normal(size=(n, 19)).astype(np.float32),
                         "actions": rng.uniform(-1, 1, size=(n, 7)).astype(np.float32)}
           for i, n in enumerate(lengths)}
    obs_n = Normalizer.fit([e["obs"] for e in eps.values()], "minmax")
    act_n = Normalizer.fit([e["actions"] for e in eps.values()], "minmax")
    return WindowTensors.from_dataset(SequenceDataset(eps, list(eps), obs_n, act_n, t_o, t_p))


CONTEXT = {"run_id": "unit", "hashes": {"observation_spec_sha256": "x"}, "git": {},
           "normalization": {"obs": {}, "action": {}, "content_sha256": "x"}}


def test_masked_mse_ignores_padded_steps() -> None:
    pred = torch.zeros(2, 4, 7)
    target = torch.ones(2, 4, 7)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
    corrupted = target.clone()
    corrupted[~valid] = 100.0
    assert float(masked_mse(pred, target, valid)) == 1.0
    assert torch.equal(masked_mse(pred, target, valid), masked_mse(pred, corrupted, valid))


def test_windows_equal_sequence_dataset_items() -> None:
    w = windows(3, 4)
    ds_items = [w.batch(np.array([i])) for i in (0, 5, len(w) - 1)]
    assert all(b["obs_history"].shape == (1, 3, 19) for b in ds_items)
    assert w.obs_valid[0].tolist() == [False, False, True]  # first step of demo_0
    assert w.valid_mask[11].tolist() == [True, False, False, False]  # last step of demo_0


def test_sampler_serves_each_window_once_per_epoch_and_resumes() -> None:
    s = EpochStreamSampler(10, 4, seed=3)
    first = np.concatenate([s.next_batch() for _ in range(5)])  # 20 = 2 epochs
    assert sorted(first[:10]) == list(range(10)) and sorted(first[10:]) == list(range(10))
    a = EpochStreamSampler(10, 4, seed=3)
    for _ in range(3):
        a.next_batch()
    state = a.state_dict()
    b = EpochStreamSampler(10, 4, seed=99)
    b.load_state_dict(state)
    assert all(np.array_equal(a.next_batch(), b.next_batch()) for _ in range(7))


@pytest.mark.parametrize(("family", "t_o"), [("bc_mlp", 1), ("bc_rnn", 6)])
def test_tiny_batch_overfit(family: str, t_o: int) -> None:
    data = windows(t_o, 1)
    small = WindowTensors(*(x[:16] for x in (data.obs_history, data.obs_valid,
                                              data.future_action, data.valid_mask)),
                          data.episode_id[:16], data.start_t[:16])
    cfg = from_dict(cfg_dict(family, t_o, 1, max_steps=300, ckpt_every=300, val_every=300))
    tr = Trainer(cfg, small, None, CONTEXT)
    first = tr.step()["loss"]
    losses = [tr.step()["loss"] for _ in range(299)]
    assert losses[-1] < 0.05 * first, (first, losses[-1])


@pytest.mark.parametrize(("family", "t_o", "t_p"), [("bc_mlp", 2, 4), ("bc_rnn", 5, 1)])
def test_resume_from_checkpoint_is_bitwise_identical(tmp_path, family, t_o, t_p) -> None:
    data = windows(t_o, t_p)
    cfg = from_dict(cfg_dict(family, t_o, t_p))
    full = Trainer(cfg, data, None, CONTEXT)
    for _ in range(40):
        full.step()
    half = Trainer(cfg, data, None, CONTEXT)
    for _ in range(20):
        half.step()
    path = tmp_path / checkpoint_name(20)
    sha = save_checkpoint(path, half.payload())
    torch.manual_seed(12345)  # disturb the global RNG: resume must not depend on it
    resumed = Trainer(cfg, data, None, CONTEXT)
    resumed.load_state(load_checkpoint(path, expected_sha256=sha))
    for _ in range(20):
        resumed.step()
    a, b = full.model.state_dict(), resumed.model.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert resumed.global_step == full.global_step == 40


def test_checkpoints_are_immutable_hash_checked_and_strict(tmp_path) -> None:
    data = windows(1, 1)
    tr = Trainer(from_dict(cfg_dict()), data, None, CONTEXT)
    path = tmp_path / "c.pt"
    sha = save_checkpoint(path, tr.payload())
    with pytest.raises(FileExistsError):
        save_checkpoint(path, tr.payload())
    with pytest.raises(CheckpointError):
        load_checkpoint(path, expected_sha256="0" * 64)
    payload = load_checkpoint(path, expected_sha256=sha)  # weights_only=True
    bad = dict(payload["model_state"])
    bad.pop(next(iter(bad)))
    with pytest.raises(RuntimeError):
        tr.model.load_state_dict(bad, strict=True)
    other = Trainer(from_dict(cfg_dict(max_steps=60)), data, None, CONTEXT)
    with pytest.raises(CheckpointError):
        other.load_state(payload)  # a different config can never resume this checkpoint


def test_non_finite_loss_raises() -> None:
    data = windows(1, 1)
    data.obs_history[0, 0, 0] = float("nan")
    cfg = from_dict(cfg_dict(batch_size=len(data)))
    with pytest.raises(FloatingPointError):
        Trainer(cfg, data, None, CONTEXT).step()


def test_config_is_strict_and_hashed() -> None:
    d = cfg_dict()
    cfg = from_dict(d)
    assert run_id(cfg, "s3") == f"s3_t_bc_mlp_lift_seed0_{cfg.sha256()[:8]}"
    assert from_dict(apply_overrides(d, ["train.max_steps=60"])).train.max_steps == 60
    assert from_dict(apply_overrides(d, ["train.max_steps=60"])).sha256() != cfg.sha256()
    with pytest.raises(KeyError):
        apply_overrides(d, ["train.max_stepz=60"])
    for broken in ({**d, "extra": 1}, {**d, "train": {**d["train"], "max_steps": 30}},
                   {**d, "horizons": {"obs": 1, "pred": 1, "exec": 2}}):
        with pytest.raises(ValueError):
            from_dict(broken)
