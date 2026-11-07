"""Diffusion noise schedule and the DDPM / DDIM reverse samplers (MASTER_SPEC §5.5).

RoboForge implements the equations itself rather than calling diffusers at run time, so that
every quantity the sampler uses can be recorded for the denoising export (``StepRecord``).
The arithmetic deliberately follows ``diffusers==0.11.1`` (the version pinned by robomimic
``d309eae``; Apache-2.0), operation for operation and in float32, and
``tests/unit/test_diffusion_reference.py`` checks it **bitwise** against that package.

Forward process (DDPM, Ho et al. 2020, eq. 4), for normalized action chunks ``A_0``::

    A_t = sqrt(abar_t) * A_0 + sqrt(1 - abar_t) * eps,   eps ~ N(0, I),  t in {0..K-1}

with ``abar_t = prod_{s<=t} (1 - beta_s)`` and the ``squaredcos_cap_v2`` (Nichol & Dhariwal
2021, "Glide cosine") betas: ``beta_i = min(1 - abar(i+1/K) / abar(i/K), 0.999)``,
``abar(u) = cos((u + 0.008) / 1.008 * pi / 2)^2``.

Reverse steps for an epsilon-predicting network ``eps_hat = f(A_t, t, c)``:

* ``x0_hat = (A_t - sqrt(1 - abar_t) * eps_hat) / sqrt(abar_t)``, clipped to [-1, 1] when
  ``clip_sample`` (the normalized action range);
* **DDPM** (ancestral, ``fixed_small`` variance): ``A_{t-1} = c0 * x0_hat + c1 * A_t + sigma_t z``
  with ``c0 = sqrt(abar_{t-1}) beta_t / (1 - abar_t)``, ``c1 = sqrt(alpha_t) (1 - abar_{t-1}) /
  (1 - abar_t)``, ``sigma_t^2 = max((1 - abar_{t-1}) / (1 - abar_t) * beta_t, 1e-20)`` and
  ``z ~ N(0, I)`` for ``t > 0`` only. diffusers 0.11.1 always uses ``t - 1`` as the previous
  timestep, so DDPM is only correct with as many inference steps as training steps; this
  module refuses a strided DDPM.
* **DDIM** (Song et al. 2021, eq. 12): timesteps ``round(i * K / N)``, ``i = N-1..0``;
  ``A_prev = sqrt(abar_prev) * x0_hat + sqrt(1 - abar_prev - sigma^2) * eps_hat (+ sigma z)``
  with ``sigma = eta * sqrt((1 - abar_prev)/(1 - abar_t) * (1 - abar_t/abar_prev))`` and
  ``abar_prev = 1`` after the last step (``set_alpha_to_one``). With ``eta = 0`` it is
  deterministic given the initial noise.

All randomness comes from an explicit ``torch.Generator``; nothing reads the global RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

BETA_SCHEDULES = ("squaredcos_cap_v2", "linear")
PREDICTION_TYPES = ("epsilon",)
SAMPLERS = ("ddpm", "ddim")


def betas_squaredcos_cap_v2(num_train_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    """Glide cosine betas, computed exactly as diffusers 0.11.1 ``betas_for_alpha_bar``."""

    def alpha_bar(u: float) -> float:
        return math.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_train_timesteps):
        t1 = i / num_train_timesteps
        t2 = (i + 1) / num_train_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class NoiseSchedule:
    """``beta_t``, ``alpha_t = 1 - beta_t`` and ``abar_t`` (float32) for ``K`` training steps."""

    def __init__(self, num_train_timesteps: int, beta_schedule: str = "squaredcos_cap_v2",
                 beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        if num_train_timesteps < 2:
            raise ValueError("need at least 2 diffusion timesteps")
        if beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_squaredcos_cap_v2(num_train_timesteps)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps,
                                        dtype=torch.float32)
        else:
            raise ValueError(f"beta_schedule {beta_schedule!r} not in {BETA_SCHEDULES}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_schedule = beta_schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Forward process ``A_t = sqrt(abar_t) A_0 + sqrt(1 - abar_t) eps`` per batch item."""
        acp = self.alphas_cumprod.to(dtype=x0.dtype, device=x0.device)
        a = (acp[timesteps] ** 0.5).flatten()
        b = ((1 - acp[timesteps]) ** 0.5).flatten()
        while a.ndim < x0.ndim:
            a, b = a.unsqueeze(-1), b.unsqueeze(-1)
        return a * x0 + b * noise

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible schedule (for manifests and the denoising export)."""
        return {"num_train_timesteps": self.num_train_timesteps,
                "beta_schedule": self.beta_schedule, "betas": self.betas.tolist(),
                "alphas_cumprod": self.alphas_cumprod.tolist()}


@dataclass
class StepRecord:
    """Everything one reverse step used and produced (for the instrumented sampler)."""

    timestep: int
    prev_timestep: int
    alpha_prod_t: float
    alpha_prod_t_prev: float
    coefficients: dict[str, float]
    sample: torch.Tensor  # A_t, the input of this step
    model_output: torch.Tensor  # eps_hat
    pred_original_unclipped: torch.Tensor  # x0_hat before clip_sample
    pred_original: torch.Tensor  # x0_hat after clip_sample (what the step used)
    noise: torch.Tensor | None  # z injected by this step, or None
    prev_sample: torch.Tensor  # A_{prev_timestep}, the output of this step


class _Sampler:
    """Shared parts of the reverse samplers."""

    name = ""

    def __init__(self, schedule: NoiseSchedule, num_inference_steps: int, clip_sample: bool,
                 prediction_type: str = "epsilon") -> None:
        if prediction_type not in PREDICTION_TYPES:
            raise ValueError(f"prediction_type {prediction_type!r} not in {PREDICTION_TYPES}")
        if not 1 <= num_inference_steps <= schedule.num_train_timesteps:
            raise ValueError("num_inference_steps must lie in [1, num_train_timesteps]")
        self.schedule = schedule
        self.num_inference_steps = int(num_inference_steps)
        self.clip_sample = bool(clip_sample)
        self.prediction_type = prediction_type
        self.timesteps: list[int] = []

    def _x0(self, model_output: torch.Tensor, t: int,
            sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_prod_t = self.schedule.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        x0 = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        clipped = torch.clamp(x0, -1, 1) if self.clip_sample else x0
        return x0, clipped, alpha_prod_t

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible sampler description."""
        return {"sampler": self.name, "num_inference_steps": self.num_inference_steps,
                "timesteps": list(self.timesteps), "clip_sample": self.clip_sample,
                "prediction_type": self.prediction_type}


class DDPMSampler(_Sampler):
    """Ancestral DDPM sampling (diffusers 0.11.1 ``DDPMScheduler``, ``fixed_small``)."""

    name = "ddpm"

    def __init__(self, schedule: NoiseSchedule, num_inference_steps: int, clip_sample: bool,
                 prediction_type: str = "epsilon") -> None:
        super().__init__(schedule, num_inference_steps, clip_sample, prediction_type)
        if num_inference_steps != schedule.num_train_timesteps:
            raise ValueError("strided DDPM is not supported: diffusers 0.11.1 always steps to "
                             "t - 1, so DDPM needs inference steps == training steps; use DDIM")
        self.timesteps = list(range(schedule.num_train_timesteps - 1, -1, -1))

    def step(self, model_output: torch.Tensor, t: int, sample: torch.Tensor,
             generator: torch.Generator) -> StepRecord:
        """One reverse step ``A_t -> A_{t-1}``."""
        s = self.schedule
        x0, x0c, alpha_prod_t = self._x0(model_output, t, sample)
        alpha_prod_t_prev = s.alphas_cumprod[t - 1] if t > 0 else s.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        c0 = (alpha_prod_t_prev ** (0.5) * s.betas[t]) / beta_prod_t
        c1 = s.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t
        prev = c0 * x0c + c1 * sample
        noise, std = None, 0.0
        if t > 0:
            noise = torch.randn(model_output.shape, generator=generator,
                                device=model_output.device, dtype=model_output.dtype)
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * s.betas[t]
            variance = torch.clamp(variance, min=1e-20)
            std = variance ** 0.5
            prev = prev + std * noise
        return StepRecord(
            timestep=int(t), prev_timestep=int(t - 1), alpha_prod_t=float(alpha_prod_t),
            alpha_prod_t_prev=float(alpha_prod_t_prev),
            coefficients={"beta_t": float(s.betas[t]), "alpha_t": float(s.alphas[t]),
                          "pred_original_coeff": float(c0), "current_sample_coeff": float(c1),
                          "noise_std": float(std)},
            sample=sample, model_output=model_output, pred_original_unclipped=x0,
            pred_original=x0c, noise=noise, prev_sample=prev)


class DDIMSampler(_Sampler):
    """DDIM sampling (diffusers 0.11.1 ``DDIMScheduler``, ``set_alpha_to_one=True``)."""

    name = "ddim"

    def __init__(self, schedule: NoiseSchedule, num_inference_steps: int, clip_sample: bool,
                 prediction_type: str = "epsilon", eta: float = 0.0,
                 steps_offset: int = 0) -> None:
        super().__init__(schedule, num_inference_steps, clip_sample, prediction_type)
        if eta < 0:
            raise ValueError("eta must be >= 0")
        self.eta, self.steps_offset = float(eta), int(steps_offset)
        self.step_ratio = schedule.num_train_timesteps // self.num_inference_steps
        ts = [round(i * self.step_ratio) for i in range(self.num_inference_steps)][::-1]
        self.timesteps = [t + self.steps_offset for t in ts]
        self.final_alpha_cumprod = torch.tensor(1.0)

    def step(self, model_output: torch.Tensor, t: int, sample: torch.Tensor,
             generator: torch.Generator) -> StepRecord:
        """One reverse step ``A_t -> A_{t - K/N}``."""
        s = self.schedule
        prev_t = t - self.step_ratio
        x0, x0c, alpha_prod_t = self._x0(model_output, t, sample)
        alpha_prod_t_prev = s.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = self.eta * variance ** (0.5)
        direction_coeff = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5)
        prev = alpha_prod_t_prev ** (0.5) * x0c + direction_coeff * model_output
        noise = None
        if self.eta > 0:
            noise = torch.randn(model_output.shape, generator=generator,
                                device=model_output.device, dtype=model_output.dtype)
            prev = prev + variance ** (0.5) * self.eta * noise
        return StepRecord(
            timestep=int(t), prev_timestep=int(prev_t), alpha_prod_t=float(alpha_prod_t),
            alpha_prod_t_prev=float(alpha_prod_t_prev),
            coefficients={"pred_original_coeff": float(alpha_prod_t_prev ** 0.5),
                          "direction_coeff": float(direction_coeff), "eta": self.eta,
                          "noise_std": float(std_dev_t)},
            sample=sample, model_output=model_output, pred_original_unclipped=x0,
            pred_original=x0c, noise=noise, prev_sample=prev)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible sampler description (adds eta and the offset)."""
        return {**super().to_dict(), "eta": self.eta, "steps_offset": self.steps_offset,
                "set_alpha_to_one": True}


def make_sampler(schedule: NoiseSchedule, sampler: str, num_inference_steps: int,
                 clip_sample: bool, prediction_type: str = "epsilon",
                 eta: float = 0.0) -> DDPMSampler | DDIMSampler:
    """The configured reverse sampler."""
    if sampler == "ddpm":
        if eta:
            raise ValueError("eta applies to DDIM only")
        return DDPMSampler(schedule, num_inference_steps, clip_sample, prediction_type)
    if sampler == "ddim":
        return DDIMSampler(schedule, num_inference_steps, clip_sample, prediction_type, eta)
    raise ValueError(f"sampler {sampler!r} not in {SAMPLERS}")
