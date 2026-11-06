"""Diffusion policy: loss masking, conditioning, seeded sampling, trainer, checkpoints.

Synthetic windows only (no dataset). Reference equivalence lives in
``test_diffusion_reference.py``.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from roboforge.models.diffusion import DiffusionConfig, DiffusionPolicy
from roboforge.models.registry import build_model, count_parameters
from roboforge.models.unet1d import SinusoidalPosEmb
from roboforge.train.checkpoint import checkpoint_name, load_checkpoint, save_checkpoint
from roboforge.train.config import from_dict, load_config
from roboforge.train.loop import Trainer, offline_metrics
from roboforge.train.run import reload_check
from tests.unit.test_train_loop import CONTEXT, windows

SMALL = {"down_dims": [16, 32], "step_embed_dim": 16, "kernel_size": 5, "n_groups": 8,
         "num_train_timesteps": 100, "beta_schedule": "squaredcos_cap_v2",
         "prediction_type": "epsilon", "clip_sample": True, "sampler": "ddim",
         "inference_steps": 10, "ddim_eta": 0.0, "ema": True, "ema_power": 0.75,
         "ema_max_decay": 0.9999}


def dcfg(**train) -> dict:
    t = {"max_steps": 40, "batch_size": 16, "ckpt_every": 20, "val_every": 20,
         "log_every": 10, "num_threads": 1, "grad_clip": None, **train}
    return {"schema": "roboforge.train_config/1", "name": "t_diffusion",
            "family": "diffusion_unet", "task": "lift", "seed": 0,
            "horizons": {"obs": 2, "pred": 16, "exec": 8}, "model": dict(SMALL),
            "loss": {"kind": "denoising_mse"},
            "optim": {"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.0,
                      "schedule": "constant"}, "train": t}


def net(**kw) -> DiffusionPolicy:
    torch.manual_seed(0)
    return build_model("diffusion_unet", {**SMALL, **kw}, 19, 7, 2, 16).eval()


def batch(n: int = 8, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    valid = torch.ones(n, 16, dtype=torch.bool)
    valid[0, 10:] = False
    return {"obs_history": torch.rand((n, 2, 19), generator=g) * 2 - 1,
            "obs_valid": torch.ones(n, 2, dtype=torch.bool),
            "future_action": torch.rand((n, 16, 7), generator=g) * 2 - 1, "valid_mask": valid}


def test_primary_config_parameter_count_and_bc_hashes_unchanged() -> None:
    cfg = load_config("configs/models/diffusion_unet.yaml")
    m = build_model(cfg.family, cfg.model, 19, 7, cfg.horizons.obs, cfg.horizons.pred)
    assert count_parameters(m) == 1_092_551
    assert sum(p.numel() for p in m.encoder.parameters()) == 0  # flatten encoder
    assert (cfg.horizons.obs, cfg.horizons.pred, cfg.horizons.exec) == (2, 16, 8)
    # Adding the diffusion family must not change any Session 3 config hash or run ID.
    for name, sha8 in (("bc_mlp", "dbf312cd"), ("bc_rnn", "2e7cffe3"),
                       ("bc_mlp_chunk", "bb8b4dd6")):
        assert load_config(f"configs/models/{name}.yaml").sha256()[:8] == sha8


def test_config_validation() -> None:
    d = dcfg()
    from_dict(d)
    for broken in ({**d, "loss": {"kind": "mse"}},
                   {**d, "model": {**SMALL, "extra": 1}},
                   {**d, "model": {**SMALL, "sampler": "ddpm", "inference_steps": 10}},
                   {**d, "model": {**SMALL, "down_dims": [8] * 6}}):
        with pytest.raises(ValueError):
            from_dict(broken)
    with pytest.raises(ValueError):
        DiffusionConfig(sampler="euler")


def test_masked_loss_ignores_padded_targets_and_uses_only_the_generator() -> None:
    m, b = net(), batch()
    rng_before = torch.get_rng_state()
    loss = m.training_loss(**b, generator=torch.Generator().manual_seed(1))
    corrupted = dict(b, future_action=b["future_action"].clone())
    corrupted["future_action"][0, 10:] = 50.0  # padded steps of window 0
    # The padded target changes A_t at those steps, so only the masked eps error is
    # compared: eps itself does not depend on the target.
    eps_hat, eps = m.noise_prediction(b["obs_history"], b["obs_valid"], b["future_action"],
                                      torch.Generator().manual_seed(1))
    eps_hat2, eps2 = m.noise_prediction(corrupted["obs_history"], corrupted["obs_valid"],
                                        corrupted["future_action"],
                                        torch.Generator().manual_seed(1))
    assert torch.equal(eps, eps2)
    from roboforge.models.losses import masked_mse

    assert torch.equal(loss, masked_mse(eps_hat, eps, b["valid_mask"]))
    padded_only = torch.zeros_like(b["valid_mask"])
    padded_only[0, 10:] = True
    w = b["valid_mask"].clone()
    w[0, 10:] = True
    assert not torch.equal(masked_mse(eps_hat, eps, w), loss)  # the mask matters
    assert torch.equal(torch.get_rng_state(), rng_before)  # global RNG untouched


def test_seeded_sampling_is_reproducible_isolated_and_bounded() -> None:
    m, b = net(), batch()
    x, v = b["obs_history"], b["obs_valid"]
    rng_before = torch.get_rng_state()
    a1 = m.sample(x, v, torch.Generator().manual_seed(5))
    a2 = m.sample(x, v, torch.Generator().manual_seed(5))
    a3 = m.sample(x, v, torch.Generator().manual_seed(6))
    assert torch.equal(a1, a2) and not torch.equal(a1, a3)
    assert a1.shape == (8, 16, 7)
    assert bool((a1.abs() <= 1).all())  # final DDIM step returns the clipped x0 estimate
    assert torch.equal(torch.get_rng_state(), rng_before)
    # Rows do not interact: sampling row 0 alone from its own initial noise reproduces row 0
    # up to float rounding (the conv arithmetic depends on the batch size; the closed-loop
    # policy always samples one row).
    noise = torch.randn((8, 16, 7), generator=torch.Generator().manual_seed(9))
    full = m.sample(x, v, torch.Generator(), initial_noise=noise)
    one = m.sample(x[:1], v[:1], torch.Generator(), initial_noise=noise[:1])
    torch.testing.assert_close(full[:1], one, rtol=0, atol=1e-4)


def test_sample_depends_on_the_condition_and_records_every_step() -> None:
    m, b = net(), batch()
    x, v = b["obs_history"], b["obs_valid"]
    steps = []
    a = m.sample(x, v, torch.Generator().manual_seed(2), trace=steps.append)
    other = m.sample(x + 0.5, v, torch.Generator().manual_seed(2))
    assert not torch.equal(a, other)
    assert [s.timestep for s in steps] == m.sampler.timesteps
    assert torch.equal(steps[-1].prev_sample, a)
    for prev, nxt in zip(steps, steps[1:], strict=False):
        assert torch.equal(prev.prev_sample, nxt.sample)
        assert nxt.timestep == prev.prev_timestep


def test_timestep_embedding() -> None:
    e = SinusoidalPosEmb(16)(torch.arange(100))
    assert e.shape == (100, 16)
    torch.testing.assert_close(e[0], torch.cat([torch.zeros(8), torch.ones(8)]))
    assert len({tuple(r.tolist()) for r in e}) == 100  # distinct per timestep


def test_gradients_reach_every_parameter() -> None:
    m, b = net(), batch()
    m.train()
    m.training_loss(**b, generator=torch.Generator().manual_seed(0)).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().sum()) > 0, name


def test_resume_is_bitwise_identical_including_noise_rng_and_ema(tmp_path) -> None:
    data = windows(2, 16)
    cfg = from_dict(dcfg())
    full = Trainer(cfg, data, None, CONTEXT)
    for _ in range(30):
        full.step()
    half = Trainer(cfg, data, None, CONTEXT)
    for _ in range(15):
        half.step()
    path = tmp_path / checkpoint_name(15)
    sha = save_checkpoint(path, half.payload())
    torch.manual_seed(999)
    resumed = Trainer(cfg, data, None, CONTEXT)
    resumed.load_state(load_checkpoint(path, expected_sha256=sha))  # weights_only
    for _ in range(15):
        resumed.step()
    for a, b in ((full.model, resumed.model), (full.inference_model, resumed.inference_model)):
        sa, sb = a.state_dict(), b.state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert torch.equal(full.noise_gen.get_state(), resumed.noise_gen.get_state())
    assert full.ema.optimization_step == resumed.ema.optimization_step == 30


def test_ema_is_the_inference_network_and_reload_check_passes(tmp_path) -> None:
    data = windows(2, 16)
    tr = Trainer(from_dict(dcfg()), data, data, CONTEXT)
    for _ in range(20):
        tr.step()
    raw, ema = tr.model.state_dict(), tr.inference_model.state_dict()
    assert any(not torch.equal(raw[k], ema[k]) for k in raw)
    path = tmp_path / "c.pt"
    sha = save_checkpoint(path, tr.payload())
    rc = reload_check(tr, path, sha)
    assert rc["passed"] and rc["inference_network"] == "ema", rc
    m = offline_metrics(tr.inference_model, data)
    assert m["mse"] is not None and np.isfinite(m["denoising_mse"])
    assert offline_metrics(tr.inference_model, data) == m  # fixed seeds: repeatable
    no_ema = Trainer(from_dict({**dcfg(), "model": {**SMALL, "ema": False}}), data, None,
                     CONTEXT)
    assert no_ema.ema is None and no_ema.inference_model is no_ema.model


def test_tiny_batch_overfit_reduces_denoising_and_sampling_error() -> None:
    data = windows(2, 16, lengths=(12, 9))
    cfg = from_dict(dcfg(max_steps=400, ckpt_every=400, val_every=400, batch_size=21))
    tr = Trainer(cfg, data, None, CONTEXT)
    before = offline_metrics(copy.deepcopy(tr.model).eval(), data)
    for _ in range(400):
        r = tr.step()
        assert np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"])
    after = offline_metrics(tr.model.eval(), data)
    # Synthetic targets are uniform noise unrelated to the observations, so only a partial
    # fit is expected here; the real-data overfit lives in tests/integration.
    assert after["denoising_mse"] < 0.5 * before["denoising_mse"], (before, after)
    assert after["mse"] < 0.8 * before["mse"], (before, after)
