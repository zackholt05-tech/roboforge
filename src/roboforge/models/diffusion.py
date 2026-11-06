"""Conditional diffusion action-chunk policy (MASTER_SPEC §5.5; RESEARCH_PROTOCOL §6.1).

:class:`DiffusionPolicy` is a real iterative denoiser of normalized action sequences, not a
deterministic head:

1. **Observation encoder** (:class:`ObservationEncoder`): the normalized ``T_o x D`` window is
   flattened into the global condition ``c`` (``T_o * D`` values). This is robomimic's
   low-dim path (its low-dim encoder concatenates the keys without learned layers) and gives
   the network exactly the input BC-MLP-chunk sees, including the pre-start padding, which
   repeats the first observation (D-022).
2. **Training** (:meth:`DiffusionPolicy.training_loss`): for the normalized target chunk
   ``A_0 [B, T_p, A]`` draw ``eps ~ N(0, I)`` and ``t ~ U{0..K-1}`` from an explicit
   generator, form ``A_t = sqrt(abar_t) A_0 + sqrt(1 - abar_t) eps``, predict
   ``eps_hat = unet(A_t, t, c)`` and take the **masked** MSE between ``eps_hat`` and ``eps``
   over valid (non-padded) action steps and all action dimensions.
3. **Sampling** (:meth:`DiffusionPolicy.sample`): start from ``A_K ~ N(0, I)``, run the
   configured DDPM or DDIM reverse steps (:mod:`roboforge.models.noise_schedule`), and return
   the final normalized chunk. Decoding to simulator actions happens outside, through
   ``ActionCodec`` (unnormalize, clip), as for every family. An optional ``trace`` callback
   receives every :class:`~roboforge.models.noise_schedule.StepRecord` of the **same** call;
   the denoising export uses it, so exported intermediates are the sampler's actual tensors.

Every random draw takes a ``torch.Generator``; the global RNG is never read, so a seeded
sample is reproducible bitwise on one platform.

**Convolution backend.** U-Net calls run inside ``torch.backends.nnpack.flags(enabled=False)``
(:func:`unet_backend`), a scoped switch that is restored on exit. On the macOS arm64 torch
build, NNPACK made a training step 3.2x slower (342 vs 109 ms, Session 4 profile) than the
default im2col path; builds that select another backend (e.g. oneDNN) are unaffected. The
same scope is used for training, sampling and the reference tests, so the arithmetic is
the same everywhere on one platform.

**Alignment with robomimic (recorded deviation).** robomimic's Diffusion Policy predicts the
chunk ``a[t-T_o+1 .. t-T_o+T_p]`` and executes it from index ``T_o - 1``. RoboForge windows
(D-022) predict ``a[t .. t+T_p-1]`` and execute from index 0. Both execute ``a[t]`` first at a
decision at frame ``t``; RoboForge keeps BC-MLP-chunk's window, as H2-c requires.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from roboforge.models.losses import masked_mse
from roboforge.models.noise_schedule import (
    BETA_SCHEDULES,
    PREDICTION_TYPES,
    SAMPLERS,
    NoiseSchedule,
    StepRecord,
    make_sampler,
)
from roboforge.models.unet1d import ConditionalUnet1D


@dataclass(frozen=True)
class DiffusionConfig:
    """Diffusion policy architecture, noise schedule, sampler and EMA (one flat section)."""

    down_dims: tuple[int, ...] = (64, 128)
    step_embed_dim: int = 64
    kernel_size: int = 5
    n_groups: int = 8
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    sampler: str = "ddim"
    inference_steps: int = 10
    ddim_eta: float = 0.0
    ema: bool = True
    ema_power: float = 0.75
    ema_max_decay: float = 0.9999

    def __post_init__(self) -> None:
        for k in ("ddim_eta", "ema_power", "ema_max_decay"):
            object.__setattr__(self, k, float(getattr(self, k)))
        object.__setattr__(self, "down_dims", tuple(int(d) for d in self.down_dims))
        if self.beta_schedule not in BETA_SCHEDULES:
            raise ValueError(f"beta_schedule {self.beta_schedule!r} not in {BETA_SCHEDULES}")
        if self.prediction_type not in PREDICTION_TYPES:
            raise ValueError(f"prediction_type {self.prediction_type!r} not in "
                             f"{PREDICTION_TYPES}")
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sampler {self.sampler!r} not in {SAMPLERS}")
        if any(d % self.n_groups for d in self.down_dims):
            raise ValueError("every down_dims width must be divisible by n_groups")
        if self.step_embed_dim < 4 or self.step_embed_dim % 2:
            raise ValueError("step_embed_dim must be even and >= 4")
        if not 0.0 <= self.ema_max_decay < 1.0:
            raise ValueError("ema_max_decay must lie in [0, 1)")
        if self.sampler == "ddpm" and self.inference_steps != self.num_train_timesteps:
            raise ValueError("strided DDPM is not supported (diffusers 0.11.1 semantics); "
                             "use DDIM for fewer inference steps")
        if not 1 <= self.inference_steps <= self.num_train_timesteps:
            raise ValueError("inference_steps must lie in [1, num_train_timesteps]")
        if self.sampler == "ddpm" and self.ddim_eta:
            raise ValueError("ddim_eta applies to the DDIM sampler only")

    def check_horizon(self, pred_horizon: int) -> None:
        """Raise unless the U-Net can resample a chunk of ``pred_horizon`` steps."""
        if pred_horizon % (2 ** (len(self.down_dims) - 1)):
            raise ValueError(f"T_p={pred_horizon} not divisible by 2^(stages-1) for "
                             f"down_dims {self.down_dims}")


@contextlib.contextmanager
def unet_backend() -> Iterator[None]:
    """Scope of every U-Net call: NNPACK convolutions disabled (see the module docstring)."""
    with torch.backends.nnpack.flags(enabled=False):
        yield


class ObservationEncoder(nn.Module):
    """Flatten the normalized observation window into the global condition (no parameters)."""

    def __init__(self, obs_dim: int, obs_horizon: int) -> None:
        super().__init__()
        self.obs_dim, self.obs_horizon = obs_dim, obs_horizon
        self.output_dim = obs_dim * obs_horizon

    def forward(self, obs_history: torch.Tensor, obs_valid: torch.Tensor) -> torch.Tensor:
        """``[B, T_o, D]`` (+ mask, which the padded window already encodes) -> ``[B, T_o*D]``."""
        if obs_history.ndim != 3 or obs_history.shape[1:] != (self.obs_horizon, self.obs_dim):
            raise ValueError(f"obs_history shape {tuple(obs_history.shape)} != [B, "
                             f"{self.obs_horizon}, {self.obs_dim}]")
        if obs_valid.shape != obs_history.shape[:2] or obs_valid.dtype != torch.bool:
            raise ValueError("obs_valid must be a bool tensor of shape [B, T_o]")
        return obs_history.flatten(start_dim=1)


TraceFn = Callable[[StepRecord], None]


class DiffusionPolicy(nn.Module):
    """Observation encoder + conditional 1D U-Net + noise schedule + reverse sampler."""

    stochastic = True

    def __init__(self, obs_dim: int, action_dim: int, obs_horizon: int, pred_horizon: int,
                 cfg: DiffusionConfig) -> None:
        super().__init__()
        cfg.check_horizon(pred_horizon)
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.obs_horizon, self.pred_horizon, self.cfg = obs_horizon, pred_horizon, cfg
        self.encoder = ObservationEncoder(obs_dim, obs_horizon)
        self.unet = ConditionalUnet1D(action_dim, self.encoder.output_dim, cfg.step_embed_dim,
                                      cfg.down_dims, cfg.kernel_size, cfg.n_groups)
        self.schedule = NoiseSchedule(cfg.num_train_timesteps, cfg.beta_schedule)
        self.sampler = make_sampler(self.schedule, cfg.sampler, cfg.inference_steps,
                                    cfg.clip_sample, cfg.prediction_type, cfg.ddim_eta)

    def training_loss(self, obs_history: torch.Tensor, obs_valid: torch.Tensor,
                      future_action: torch.Tensor, valid_mask: torch.Tensor,
                      generator: torch.Generator) -> torch.Tensor:
        """Masked epsilon-prediction MSE for one batch (noise, then timesteps, are drawn)."""
        eps_hat, eps = self.noise_prediction(obs_history, obs_valid, future_action, generator)
        return masked_mse(eps_hat, eps, valid_mask)

    def noise_prediction(self, obs_history: torch.Tensor, obs_valid: torch.Tensor,
                         future_action: torch.Tensor,
                         generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``eps`` then ``t`` (in that order, as robomimic), noise ``A_0`` and return
        ``(eps_hat, eps)``."""
        if future_action.shape[1:] != (self.pred_horizon, self.action_dim):
            raise ValueError(f"future_action shape {tuple(future_action.shape)} != [B, "
                             f"{self.pred_horizon}, {self.action_dim}]")
        cond = self.encoder(obs_history, obs_valid)
        noise = torch.randn(future_action.shape, generator=generator,
                            dtype=future_action.dtype)
        t = torch.randint(0, self.schedule.num_train_timesteps, (future_action.shape[0],),
                          generator=generator).long()
        noisy = self.schedule.add_noise(future_action, noise, t)
        with unet_backend():
            eps_hat = self.unet(noisy, t, cond)
        return eps_hat, noise

    @torch.no_grad()
    def sample(self, obs_history: torch.Tensor, obs_valid: torch.Tensor,
               generator: torch.Generator, trace: TraceFn | None = None,
               initial_noise: torch.Tensor | None = None) -> torch.Tensor:
        """Seeded reverse diffusion -> normalized action chunk ``[B, T_p, A]``.

        ``initial_noise`` replaces the first draw ``A_K ~ N(0, I)`` (reference tests only).
        """
        cond = self.encoder(obs_history, obs_valid)
        shape = (obs_history.shape[0], self.pred_horizon, self.action_dim)
        x = (torch.randn(shape, generator=generator, dtype=obs_history.dtype)
             if initial_noise is None else initial_noise)
        for t in self.sampler.timesteps:
            with unet_backend():
                eps = self.unet(x, torch.tensor(t, dtype=torch.int64), cond)
            rec = self.sampler.step(eps, t, x, generator)
            if trace is not None:
                trace(rec)
            x = rec.prev_sample
        return x

    def forward(self, obs_history: torch.Tensor, obs_valid: torch.Tensor,
                generator: torch.Generator) -> torch.Tensor:
        """Same as :meth:`sample` (an explicit generator is required)."""
        return self.sample(obs_history, obs_valid, generator)

    def describe(self) -> dict[str, Any]:
        """Schedule and sampler description for manifests and exports."""
        return {"schedule": {"num_train_timesteps": self.schedule.num_train_timesteps,
                             "beta_schedule": self.schedule.beta_schedule},
                "sampler": self.sampler.to_dict(), "prediction_type": self.cfg.prediction_type,
                "observation_encoder": "flatten of the normalized T_o x D window "
                                       "(no parameters)"}
