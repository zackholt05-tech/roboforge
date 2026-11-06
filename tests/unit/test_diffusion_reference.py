"""RoboForge diffusion components vs the pinned references (MASTER_SPEC §5.5, Session 4).

References: ``diffusers==0.11.1`` schedulers and ``EMAModel`` (Apache-2.0) and robomimic
``d309eae`` Diffusion Policy (``ConditionalUnet1D`` and ``DiffusionPolicyUNet``, MIT), both
installed from ``uv.lock``. Comparisons use fixed synthetic arrays and are **bitwise**
unless a reduction order legitimately differs (stated per test).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from diffusers.training_utils import EMAModel

from roboforge.models.diffusion import unet_backend
from roboforge.models.noise_schedule import DDIMSampler, DDPMSampler, NoiseSchedule
from roboforge.models.reference import (
    TO,
    TP,
    A,
    D,
    K,
    ours_from,
    ref_ddim,
    ref_ddpm,
    ref_unet,
    robomimic_algo,
)
from roboforge.models.unet1d import ConditionalUnet1D
from roboforge.train.ema import EMA

B = 4


def test_schedule_equals_diffusers_bitwise() -> None:
    s, ref = NoiseSchedule(K, "squaredcos_cap_v2"), ref_ddpm()
    assert torch.equal(s.betas, ref.betas)
    assert torch.equal(s.alphas, ref.alphas)
    assert torch.equal(s.alphas_cumprod, ref.alphas_cumprod)
    assert torch.equal(s.alphas_cumprod, ref_ddim().alphas_cumprod)
    acp = s.alphas_cumprod
    assert bool((acp[1:] < acp[:-1]).all()) and float(acp[-1]) < 1e-3 < float(acp[0])


def test_add_noise_equals_diffusers_and_closed_form() -> None:
    g = torch.Generator().manual_seed(1)
    x0 = torch.rand((B, TP, A), generator=g) * 2 - 1
    eps = torch.randn((B, TP, A), generator=g)
    t = torch.tensor([0, 17, 63, 99])
    s = NoiseSchedule(K)
    got = s.add_noise(x0, eps, t)
    assert torch.equal(got, ref_ddpm().add_noise(x0, eps, t))
    a = s.alphas_cumprod[t].view(-1, 1, 1)
    torch.testing.assert_close(got, a.sqrt() * x0 + (1 - a).sqrt() * eps, rtol=0, atol=1e-6)


def test_ddpm_steps_equal_diffusers_bitwise() -> None:
    s, ref = NoiseSchedule(K), ref_ddpm()
    ref.set_timesteps(K)
    ours = DDPMSampler(s, K, clip_sample=True)
    assert ours.timesteps == ref.timesteps.tolist() == list(range(K - 1, -1, -1))
    g = torch.Generator().manual_seed(2)
    x = torch.randn((B, TP, A), generator=g)
    eps = torch.randn((B, TP, A), generator=g) * 3  # large enough to exercise clip_sample
    for t in (99, 50, 1, 0):
        rec = ours.step(eps, t, x, torch.Generator().manual_seed(t))
        out = ref.step(eps, t, x, generator=torch.Generator().manual_seed(t))
        assert torch.equal(rec.prev_sample, out.prev_sample), t
        assert torch.equal(rec.pred_original, out.pred_original_sample), t
        assert (rec.noise is None) == (t == 0)
    assert bool((ours.step(eps, 99, x, g).pred_original.abs() <= 1).all())


def test_ddim_steps_equal_diffusers_bitwise() -> None:
    s, ref = NoiseSchedule(K), ref_ddim()
    ref.set_timesteps(10)
    ours = DDIMSampler(s, 10, clip_sample=True)
    assert ours.timesteps == ref.timesteps.tolist() == [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    g = torch.Generator().manual_seed(3)
    x = torch.randn((B, TP, A), generator=g)
    for t in ours.timesteps:
        eps = torch.randn((B, TP, A), generator=g) * 2
        rec = ours.step(eps, t, x, g)
        out = ref.step(eps, t, x)
        assert torch.equal(rec.prev_sample, out.prev_sample), t
        assert torch.equal(rec.pred_original, out.pred_original_sample), t
        x = rec.prev_sample
    assert rec.prev_timestep == -10 and rec.alpha_prod_t_prev == 1.0
    assert torch.equal(rec.prev_sample, rec.pred_original)  # final step returns clipped x0


def test_strided_ddpm_is_refused() -> None:
    with pytest.raises(ValueError, match="strided DDPM"):
        DDPMSampler(NoiseSchedule(K), 10, clip_sample=True)


def test_oracle_noise_is_inverted_exactly() -> None:
    """Known reconstruction: with the true eps, x0_hat equals A_0 at every t, and one DDIM
    step to t_prev lands exactly on the forward-process point of the same eps."""
    s = NoiseSchedule(K)
    g = torch.Generator().manual_seed(4)
    x0 = torch.rand((B, TP, A), generator=g) * 1.8 - 0.9
    eps = torch.randn((B, TP, A), generator=g)
    ddim = DDIMSampler(s, 10, clip_sample=True)
    for t in ddim.timesteps:
        xt = s.add_noise(x0, eps, torch.full((B,), t))
        rec = ddim.step(eps, t, xt, g)
        torch.testing.assert_close(rec.pred_original, x0, rtol=0, atol=2e-4)
        expect = (x0 if rec.prev_timestep < 0
                  else s.add_noise(x0, eps, torch.full((B,), rec.prev_timestep)))
        torch.testing.assert_close(rec.prev_sample, expect, rtol=0, atol=2e-4)


def test_unet_loads_robomimic_weights_and_matches_bitwise() -> None:
    for down, dsed in (((64, 128), 64), ((32, 64, 128), 32)):
        ref = ref_unet(down, dsed)
        ours = ConditionalUnet1D(A, TO * D, dsed, down, 5, 8)
        ours.load_state_dict(ref.state_dict(), strict=True)
        assert sum(p.numel() for p in ours.parameters()) == sum(
            p.numel() for p in ref.parameters())
        g = torch.Generator().manual_seed(5)
        x = torch.randn((B, TP, A), generator=g)
        c = torch.randn((B, TO * D), generator=g)
        t = torch.tensor([0, 5, 50, 99])
        with unet_backend(), torch.no_grad():
            assert torch.equal(ours(x, t, c), ref(x, t, c))
            assert torch.equal(ours(x, torch.tensor(7), c), ref(x, torch.tensor(7), c))
        assert ours(x, t, c).shape == (B, TP, A)


@pytest.mark.parametrize(("sampler", "steps"), [("ddim", 10), ("ddpm", 100)])
def test_sampling_matches_robomimic_get_action_trajectory(sampler, steps) -> None:
    """Same weights, same seed: robomimic's rollout sampler (global RNG, diffusers
    scheduler) and RoboForge's (explicit generator) give bitwise-identical chunks."""
    algo = robomimic_algo(sampler, steps)
    algo.nets.eval()
    ours = ours_from(algo, sampler, steps)
    obs = torch.rand((B, TO, D), generator=torch.Generator().manual_seed(6)) * 2 - 1
    torch.manual_seed(123)
    with unet_backend():
        ref = algo._get_action_trajectory({"obs": obs.clone()})
    got = ours.sample(obs, torch.ones(B, TO, dtype=torch.bool),
                      torch.Generator().manual_seed(123))
    # robomimic returns the executed slice [T_o - 1, T_o - 1 + T_a) of the same chunk
    assert torch.equal(got[:, TO - 1:TO - 1 + 8], ref)


def test_training_loss_matches_robomimic_train_on_batch() -> None:
    """Same weights and seed: identical noise and timesteps draws; the loss agrees to float
    rounding (robomimic: F.mse_loss mean; RoboForge: masked sum / count, all steps valid)."""
    algo = robomimic_algo("ddpm", 100)
    ours = ours_from(algo, "ddpm", 100).train()
    g = torch.Generator().manual_seed(7)
    batch = {"obs": {"obs": torch.rand((B, TO, D), generator=g) * 2 - 1},
             "actions": torch.rand((B, TP, A), generator=g) * 2 - 1, "goal_obs": None}
    torch.manual_seed(321)
    with unet_backend():
        info = algo.train_on_batch(batch, epoch=0, validate=True)
    loss = ours.training_loss(batch["obs"]["obs"], torch.ones(B, TO, dtype=torch.bool),
                              batch["actions"], torch.ones(B, TP, dtype=torch.bool),
                              torch.Generator().manual_seed(321))
    torch.testing.assert_close(loss, info["losses"]["l2_loss"], rtol=1e-6, atol=0)


def test_ema_matches_diffusers_bitwise() -> None:
    torch.manual_seed(8)
    model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.GroupNorm(1, 7),
                                torch.nn.Linear(7, 3))
    ours, ref = EMA(model, power=0.75), EMAModel(model=model, power=0.75)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    for _ in range(25):
        loss = model(torch.randn(16, 5)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ours.step(model)
        ref.step(model)
        assert ours.decay == ref.decay
    sa, sb = ours.averaged_model.state_dict(), ref.averaged_model.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert ours.get_decay(0) == ours.get_decay(1) == 0.0
    np.testing.assert_allclose(ours.get_decay(10_001), 1 - 10_001 ** -0.75)
    assert ours.get_decay(10 ** 9) == 0.9999
