"""Compare RoboForge's diffusion policy with the pinned references on fixed synthetic arrays.

    ROBOFORGE_EVIDENCE_SESSION=4 python scripts/compare_diffusion_reference.py

References: ``diffusers==0.11.1`` DDPM/DDIM schedulers and ``EMAModel`` (Apache-2.0), and
robomimic ``d309eae`` Diffusion Policy (``ConditionalUnet1D``, ``DiffusionPolicyUNet``; MIT).
The same checks run as unit tests (``tests/unit/test_diffusion_reference.py``); this script
records the measured differences, tensor shapes and seeded-reproducibility results as
evidence in ``<out>`` (default
``reports/session$ROBOFORGE_EVIDENCE_SESSION/reference/diffusion_reference_<platform>.json``).
Every comparison except the training loss is expected to be bitwise (difference 0.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers.training_utils import EMAModel

from roboforge.models.diffusion import DiffusionConfig, DiffusionPolicy, unet_backend
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
from roboforge.paths import evidence_reports_dir, repo_relative
from roboforge.provenance import environment_snapshot, evaluation_platform, git_revision
from roboforge.train.ema import EMA

B = 4


def diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Max absolute difference, bitwise flag and shapes."""
    return {"max_abs_diff": float((a.double() - b.double()).abs().max()),
            "bitwise_equal": bool(torch.equal(a, b)), "shape": list(a.shape),
            "reference_shape": list(b.shape)}


def schedule_checks() -> dict[str, Any]:
    s, ddpm, ddim = NoiseSchedule(K), ref_ddpm(), ref_ddim()
    out: dict[str, Any] = {"betas": diff(s.betas, ddpm.betas),
                           "alphas_cumprod": diff(s.alphas_cumprod, ddpm.alphas_cumprod),
                           "alphas_cumprod_vs_ddim": diff(s.alphas_cumprod, ddim.alphas_cumprod),
                           "alpha_bar_first_last": [float(s.alphas_cumprod[0]),
                                                    float(s.alphas_cumprod[-1])]}
    g = torch.Generator().manual_seed(1)
    x0 = torch.rand((B, TP, A), generator=g) * 2 - 1
    eps = torch.randn((B, TP, A), generator=g)
    t = torch.tensor([0, 17, 63, 99])
    out["add_noise"] = diff(s.add_noise(x0, eps, t), ddpm.add_noise(x0, eps, t))
    ddpm.set_timesteps(K)
    ours = DDPMSampler(s, K, clip_sample=True)
    x = torch.randn((B, TP, A), generator=g)
    e3 = torch.randn((B, TP, A), generator=g) * 3
    worst = 0.0
    for tt in ours.timesteps:
        a = ours.step(e3, tt, x, torch.Generator().manual_seed(tt)).prev_sample
        b = ddpm.step(e3, tt, x, generator=torch.Generator().manual_seed(tt)).prev_sample
        worst = max(worst, float((a - b).abs().max()))
    out["ddpm_step_all_100_timesteps_max_abs_diff"] = worst
    ddim.set_timesteps(10)
    od = DDIMSampler(s, 10, clip_sample=True)
    out["ddim_timesteps"] = {"ours": od.timesteps, "reference": ddim.timesteps.tolist()}
    worst = 0.0
    for tt in od.timesteps:
        worst = max(worst, float((od.step(e3, tt, x, g).prev_sample
                                  - ddim.step(e3, tt, x).prev_sample).abs().max()))
    out["ddim_step_all_10_timesteps_max_abs_diff"] = worst
    return out


def unet_checks() -> dict[str, Any]:
    out = {}
    for down, dsed in (((64, 128), 64), ((32, 64, 128), 32), ((256, 512, 1024), 256)):
        ref = ref_unet(down, dsed)
        ours = ConditionalUnet1D(A, TO * D, dsed, down, 5, 8)
        ours.load_state_dict(ref.state_dict(), strict=True)
        g = torch.Generator().manual_seed(5)
        x, c = torch.randn((B, TP, A), generator=g), torch.randn((B, TO * D), generator=g)
        t = torch.tensor([0, 5, 50, 99])
        with unet_backend(), torch.no_grad():
            out[f"{list(down)}_embed{dsed}"] = {
                **diff(ours(x, t, c), ref(x, t, c)),
                "params": sum(p.numel() for p in ours.parameters()),
                "reference_params": sum(p.numel() for p in ref.parameters()),
                "state_dict_keys_equal": list(ours.state_dict()) == list(ref.state_dict())}
    return out


def algo_checks() -> dict[str, Any]:
    out: dict[str, Any] = {}
    obs = torch.rand((B, TO, D), generator=torch.Generator().manual_seed(6)) * 2 - 1
    for sampler, steps in (("ddim", 10), ("ddpm", 100)):
        algo = robomimic_algo(sampler, steps)
        algo.nets.eval()
        ours = ours_from(algo, sampler, steps)
        torch.manual_seed(123)
        with unet_backend():
            ref = algo._get_action_trajectory({"obs": obs.clone()})
        got = ours.sample(obs, torch.ones(B, TO, dtype=torch.bool),
                          torch.Generator().manual_seed(123))
        again = ours.sample(obs, torch.ones(B, TO, dtype=torch.bool),
                            torch.Generator().manual_seed(123))
        other = ours.sample(obs, torch.ones(B, TO, dtype=torch.bool),
                            torch.Generator().manual_seed(124))
        out[f"sampling_{sampler}{steps}"] = {
            **diff(got[:, TO - 1:TO - 1 + 8], ref),
            "note": "robomimic returns the executed slice [T_o-1, T_o-1+T_a) of the sampled "
                    "chunk; RoboForge's full chunk is [B, T_p, A]",
            "full_chunk_shape": list(got.shape),
            "same_seed_bitwise_reproducible": bool(torch.equal(got, again)),
            "different_seed_differs": not bool(torch.equal(got, other))}
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
    ref_loss, loss = info["losses"]["l2_loss"].detach(), loss.detach()
    out["training_loss"] = {"ours": float(loss), "robomimic_train_on_batch": float(ref_loss),
                            "abs_diff": float(abs(loss - ref_loss)),
                            "note": "same noise and timestep draws; reduction differs "
                                    "(masked sum/count vs F.mse_loss mean), all steps valid"}
    return out


def ema_checks() -> dict[str, Any]:
    torch.manual_seed(8)
    model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.GroupNorm(1, 7),
                                torch.nn.Linear(7, 3))
    ours, ref = EMA(model, power=0.75), EMAModel(model=model, power=0.75)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    decays = []
    for _ in range(25):
        loss = model(torch.randn(16, 5)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ours.step(model)
        ref.step(model)
        decays.append((ours.decay, ref.decay))
    sa, sb = ours.averaged_model.state_dict(), ref.averaged_model.state_dict()
    return {"steps": 25, "decays_equal": all(a == b for a, b in decays),
            "weights_bitwise_equal": all(torch.equal(sa[k], sb[k]) for k in sa),
            "decay_after_25": decays[-1][0]}


def shapes() -> dict[str, Any]:
    net = DiffusionPolicy(D, A, TO, TP, DiffusionConfig())
    oh, ov = torch.zeros(3, TO, D), torch.ones(3, TO, dtype=torch.bool)
    steps = []
    chunk = net.sample(oh, ov, torch.Generator().manual_seed(0), trace=steps.append)
    return {"obs_history": [3, TO, D], "global_condition": list(net.encoder(oh, ov).shape),
            "sample": list(chunk.shape), "trace_steps": len(steps),
            "trace_timesteps": [s.timestep for s in steps],
            "noise_prediction": list(net.unet(chunk, torch.tensor(5),
                                              net.encoder(oh, ov)).shape),
            "params": sum(p.numel() for p in net.parameters())}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    platform = evaluation_platform()
    out = args.out or evidence_reports_dir(
        "reference", f"diffusion_reference_{platform['platform_id']}.json")
    report = {"schema": "roboforge.diffusion_reference/1", "platform": platform,
              "git": git_revision(), "environment": environment_snapshot(),
              "references": {"diffusers": "0.11.1 (Apache-2.0)",
                             "robomimic": "d309eae diffusion_policy (MIT)"},
              "schedule": schedule_checks(), "unet": unet_checks(), "algo": algo_checks(),
              "ema": ema_checks(), "shapes": shapes()}
    bitwise = ([report["schedule"][k]["bitwise_equal"] for k in
                ("betas", "alphas_cumprod", "add_noise")]
               + [report["schedule"]["ddpm_step_all_100_timesteps_max_abs_diff"] == 0.0,
                  report["schedule"]["ddim_step_all_10_timesteps_max_abs_diff"] == 0.0]
               + [v["bitwise_equal"] for v in report["unet"].values()]
               + [report["algo"][k]["bitwise_equal"] for k in ("sampling_ddim10",
                                                               "sampling_ddpm100")]
               + [report["ema"]["weights_bitwise_equal"]])
    report["passed"] = bool(all(bitwise) and report["algo"]["training_loss"]["abs_diff"] < 1e-6)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"bitwise checks {sum(bitwise)}/{len(bitwise)}; training-loss |diff| "
          f"{report['algo']['training_loss']['abs_diff']:.2e} -> "
          f"{'PASS' if report['passed'] else 'FAIL'} ({repo_relative(out)})")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
