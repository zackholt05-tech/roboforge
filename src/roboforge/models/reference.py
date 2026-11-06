"""Builders for the pinned external references of the diffusion policy (tests and evidence).

Used by ``tests/unit/test_diffusion_reference.py`` and ``scripts/compare_diffusion_reference.py``
only; no training or evaluation path imports this module. The references are the
``diffusers==0.11.1`` schedulers (Apache-2.0) and robomimic ``d309eae`` Diffusion Policy
(MIT), both installed from ``uv.lock``.

robomimic ``d309eae`` builds ``ConditionalUnet1D(input_dim, global_cond_dim)`` with its default
widths (256/512/1024, 65.8 M parameters for this task) whatever ``algo.unet.*`` says, so
:func:`robomimic_algo` swaps in a robomimic U-Net of RoboForge's widths (and rebuilds the EMA
around it) after construction.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from roboforge.models.diffusion import DiffusionConfig, DiffusionPolicy

K, TO, TP, A, D = 100, 2, 16, 7, 19


def ref_ddpm() -> Any:
    """diffusers DDPMScheduler with robomimic d309eae's DDPM settings."""
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    return DDPMScheduler(num_train_timesteps=K, beta_schedule="squaredcos_cap_v2",
                         clip_sample=True, prediction_type="epsilon")


def ref_ddim() -> Any:
    """diffusers DDIMScheduler with robomimic d309eae's DDIM settings."""
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    return DDIMScheduler(num_train_timesteps=K, beta_schedule="squaredcos_cap_v2",
                         clip_sample=True, set_alpha_to_one=True, steps_offset=0,
                         prediction_type="epsilon")


def ref_unet(down: tuple[int, ...] = (64, 128), dsed: int = 64, seed: int = 0) -> Any:
    """robomimic's ConditionalUnet1D (seeded initialization)."""
    from robomimic.models.diffusion_policy_nets import ConditionalUnet1D as RefUnet

    torch.manual_seed(seed)
    return RefUnet(input_dim=A, global_cond_dim=TO * D, diffusion_step_embed_dim=dsed,
                   down_dims=list(down), kernel_size=5, n_groups=8)


def robomimic_algo(sampler: str, steps: int) -> Any:
    """robomimic's DiffusionPolicyUNet algo object with a 64/128 U-Net swapped in."""
    import robomimic.utils.obs_utils as ObsUtils
    from diffusers.training_utils import EMAModel
    from robomimic.algo import algo_factory
    from robomimic.config import config_factory

    cfg = config_factory("diffusion_policy")
    with cfg.unlocked():
        cfg.observation.modalities.obs.low_dim = ["obs"]
        cfg.observation.modalities.obs.rgb = []
        # robomimic's train script fills these in before building the cosine LR schedule
        cfg.algo.optim_params.policy.num_train_batches = 10
        cfg.algo.optim_params.policy.num_epochs = 1
        cfg.algo.ddpm.enabled = sampler == "ddpm"
        cfg.algo.ddim.enabled = sampler == "ddim"
        cfg.algo.ddpm.num_inference_timesteps = steps
        cfg.algo.ddim.num_inference_timesteps = steps
    ObsUtils.initialize_obs_utils_with_config(cfg)
    algo = algo_factory(cfg.algo_name, cfg, obs_key_shapes=OrderedDict(obs=[D]), ac_dim=A,
                        device=torch.device("cpu"))
    algo.nets["policy"]["noise_pred_net"] = ref_unet()
    algo.ema = EMAModel(model=algo.nets, power=cfg.algo.ema.power)
    return algo


def ours_from(algo: Any, sampler: str, steps: int) -> DiffusionPolicy:
    """A RoboForge DiffusionPolicy carrying the robomimic algo's U-Net weights."""
    net = DiffusionPolicy(D, A, TO, TP, DiffusionConfig(sampler=sampler, inference_steps=steps))
    net.unet.load_state_dict(algo.nets["policy"]["noise_pred_net"].state_dict(), strict=True)
    return net.eval()
