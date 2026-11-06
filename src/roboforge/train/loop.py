"""The shared supervised trainer for every policy family (RESEARCH_PROTOCOL §6.2).

One :class:`Trainer` step:

1. draw the next batch of window indices from the seeded sampler;
2. compute the family's loss, masked so that padded action steps carry no weight:
   * BC: forward ``(obs_history, obs_valid)`` and take the masked MSE against
     ``future_action``;
   * diffusion: draw noise and diffusion timesteps from the run's own noise generator and
     take the masked noise-prediction MSE (:meth:`DiffusionPolicy.training_loss`);
3. backward, then take the global gradient norm (clipped only if ``grad_clip`` is set);
4. take the optimizer and scheduler step, then the EMA update if the family uses one.

A non-finite loss or gradient norm raises instead of being skipped. Every RNG the step uses
(torch for initialization and dropout, the sampler's numpy generator, the diffusion noise
generator) and the EMA state are captured in checkpoints, so :meth:`Trainer.load_state`
followed by more steps is bitwise identical to uninterrupted training on the same platform
and thread count.

The **inference network** (:attr:`Trainer.inference_model`) is the EMA copy when the family
uses EMA, otherwise the trained network. Offline metrics, reload checks and closed-loop
policies all use it.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from roboforge.models.losses import masked_mse
from roboforge.models.registry import build_model, count_parameters, is_stochastic
from roboforge.train.checkpoint import SCHEMA, CheckpointError
from roboforge.train.config import RunConfig
from roboforge.train.data import EpochStreamSampler, WindowTensors
from roboforge.train.ema import EMA

# Fixed seeds of the offline diffusion metrics: the same noise, timesteps and initial samples
# at every evaluation, so values are comparable across steps (never used for selection).
OFFLINE_DENOISE_SEED = 0
OFFLINE_SAMPLE_SEED = 1


def noise_generator_seed(seed: int) -> int:
    """Seed of a run's diffusion training-noise generator (independent of initialization)."""
    return int(np.random.SeedSequence([int(seed), 2]).generate_state(1, np.uint64)[0])


@torch.no_grad()
def diffusion_offline_metrics(model: torch.nn.Module, data: WindowTensors, *, sample: bool,
                              batch_size: int = 1024) -> dict[str, Any]:
    """Masked noise-prediction MSE with fixed draws and, if ``sample``, the masked MSE of
    seeded sampled chunks against the demonstrated chunks (normalized space)."""
    g_noise = torch.Generator().manual_seed(OFFLINE_DENOISE_SEED)
    g_sample = torch.Generator().manual_seed(OFFLINE_SAMPLE_SEED)
    a_dim = data.future_action.shape[-1]
    den = torch.zeros(a_dim, dtype=torch.float64)
    act = torch.zeros(a_dim, dtype=torch.float64)
    n_valid = 0.0
    for start in range(0, len(data), batch_size):
        b = data.batch(np.arange(start, min(start + batch_size, len(data))))
        m = b["valid_mask"].to(torch.float64).unsqueeze(-1)
        eps_hat, eps = model.noise_prediction(b["obs_history"], b["obs_valid"],
                                              b["future_action"], g_noise)
        den += (((eps_hat.double() - eps.double()) ** 2) * m).sum(dim=(0, 1))
        if sample:
            chunk = model.sample(b["obs_history"], b["obs_valid"], g_sample)
            act += (((chunk.double() - b["future_action"].double()) ** 2) * m).sum(dim=(0, 1))
        n_valid += float(m.sum())
    out = {"denoising_mse": float(den.sum() / (n_valid * a_dim)),
           "denoising_mse_per_dim": (den / n_valid).tolist(),
           "mse": float(act.sum() / (n_valid * a_dim)) if sample else None,
           "mse_per_dim": (act / n_valid).tolist() if sample else None,
           "mse_kind": "sampled chunk vs demonstrated chunk" if sample else None,
           "n_windows": len(data), "n_valid_steps": int(n_valid),
           "seeds": {"denoise": OFFLINE_DENOISE_SEED, "sample": OFFLINE_SAMPLE_SEED}}
    return out


@torch.no_grad()
def offline_metrics(model: torch.nn.Module, data: WindowTensors, batch_size: int = 2048,
                    sample: bool = True) -> dict[str, Any]:
    """Exact masked MSE over all windows of a split, overall and per action dimension.

    For diffusion networks this is :func:`diffusion_offline_metrics` (``sample`` selects
    whether seeded chunks are also sampled)."""
    if is_stochastic(model):
        return diffusion_offline_metrics(model, data, sample=sample)
    was_training = model.training
    model.eval()
    sq = torch.zeros(data.future_action.shape[-1], dtype=torch.float64)
    n_valid = 0.0
    for start in range(0, len(data), batch_size):
        b = data.batch(np.arange(start, min(start + batch_size, len(data))))
        pred = model(b["obs_history"], b["obs_valid"])
        m = b["valid_mask"].to(torch.float64).unsqueeze(-1)
        sq += (((pred.double() - b["future_action"].double()) ** 2) * m).sum(dim=(0, 1))
        n_valid += float(m.sum())
    model.train(was_training)
    per_dim = (sq / n_valid).tolist()
    return {"mse": float(sq.sum() / (n_valid * len(per_dim))), "mse_per_dim": per_dim,
            "n_windows": len(data), "n_valid_steps": int(n_valid)}


def _scheduler(opt: torch.optim.Optimizer, cfg: RunConfig) -> Any:
    if cfg.optim.schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.max_steps)


class Trainer:
    """Model, optimizer, scheduler and sampler for one run, with exact checkpoint state."""

    def __init__(self, cfg: RunConfig, train: WindowTensors, valid: WindowTensors | None,
                 context: dict[str, Any]) -> None:
        torch.set_num_threads(cfg.train.num_threads)
        torch.manual_seed(cfg.seed)
        self.cfg, self.train_data, self.valid_data, self.context = cfg, train, valid, context
        _, t_o, obs_dim = train.obs_history.shape
        _, t_p, action_dim = train.future_action.shape
        if (t_o, t_p) != (cfg.horizons.obs, cfg.horizons.pred):
            raise ValueError(f"windows are T_o={t_o}, T_p={t_p}; config says {cfg.horizons}")
        self.dims = {"obs_dim": int(obs_dim), "action_dim": int(action_dim),
                     "obs_horizon": cfg.horizons.obs, "pred_horizon": cfg.horizons.pred,
                     "exec_horizon": cfg.horizons.exec}
        self.model = build_model(cfg.family, cfg.model, obs_dim, action_dim, t_o, t_p)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=cfg.optim.lr, betas=cfg.optim.betas,
                                     weight_decay=cfg.optim.weight_decay)
        self.sched = _scheduler(self.opt, cfg)
        self.sampler = EpochStreamSampler(len(train), cfg.train.batch_size, cfg.seed)
        self.stochastic = is_stochastic(self.model)
        self.noise_gen: torch.Generator | None = None
        self.ema: EMA | None = None
        if self.stochastic:
            self.noise_gen = torch.Generator().manual_seed(noise_generator_seed(cfg.seed))
            if cfg.model["ema"]:
                self.ema = EMA(self.model, power=cfg.model["ema_power"],
                               max_value=cfg.model["ema_max_decay"])
        self.global_step = 0

    @property
    def inference_model(self) -> torch.nn.Module:
        """The network that is evaluated: the EMA copy if the family uses EMA."""
        return self.ema.averaged_model if self.ema is not None else self.model

    @property
    def param_count(self) -> int:
        """Exact trainable parameter count."""
        return count_parameters(self.model)

    def step(self) -> dict[str, float]:
        """One optimizer step; returns loss, gradient norm and learning rate."""
        b = self.train_data.batch(self.sampler.next_batch())
        self.model.train()
        if self.stochastic:
            loss = self.model.training_loss(b["obs_history"], b["obs_valid"],
                                            b["future_action"], b["valid_mask"],
                                            generator=self.noise_gen)
        else:
            pred = self.model(b["obs_history"], b["obs_valid"])
            loss = masked_mse(pred, b["future_action"], b["valid_mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {float(loss.detach())} at step "
                                     f"{self.global_step + 1}")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in self.params]
        if any(g is None for g in grads):
            raise RuntimeError("a trainable parameter received no gradient")
        if self.cfg.train.grad_clip is None:
            gnorm = torch.nn.utils.get_total_norm(grads)
        else:
            gnorm = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.train.grad_clip)
        if not torch.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient norm at step {self.global_step + 1}")
        lr = float(self.opt.param_groups[0]["lr"])
        self.opt.step()
        self.sched.step()
        if self.ema is not None:
            self.ema.step(self.model)
        self.global_step += 1
        return {"loss": float(loss.detach()), "grad_norm": float(gnorm.detach()), "lr": lr}

    def payload(self) -> dict[str, Any]:
        """Complete checkpoint payload at the current step (diffusion adds the noise
        generator state and, with EMA, the EMA state)."""
        extra: dict[str, Any] = {}
        if self.noise_gen is not None:
            extra["noise_rng_state"] = self.noise_gen.get_state()
        if self.ema is not None:
            extra["ema_state"] = self.ema.state_dict()
        return {**extra, "schema": SCHEMA, "run_id": self.context["run_id"],
                "family": self.cfg.family, "task": self.cfg.task, "global_step": self.global_step,
                "config": self.cfg.to_dict(), "config_sha256": self.cfg.sha256(),
                "dims": dict(self.dims), "param_count": self.param_count,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.opt.state_dict(),
                "scheduler_state": self.sched.state_dict(),
                "sampler_state": self.sampler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "normalization": self.context["normalization"],
                "hashes": self.context["hashes"], "git": self.context["git"],
                "torch_version": str(torch.__version__), "device": "cpu",
                "num_threads": torch.get_num_threads()}

    def load_state(self, payload: dict[str, Any]) -> None:
        """Restore a checkpoint of the same config for exact resumption (strict)."""
        if payload["config_sha256"] != self.cfg.sha256():
            raise CheckpointError("checkpoint was trained with a different config")
        if payload["dims"] != self.dims:
            raise CheckpointError(f"checkpoint dims {payload['dims']} != {self.dims}")
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.opt.load_state_dict(payload["optimizer_state"])
        self.sched.load_state_dict(payload["scheduler_state"])
        self.sampler.load_state_dict(payload["sampler_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if self.noise_gen is not None:
            self.noise_gen.set_state(payload["noise_rng_state"])
        if self.ema is not None:
            self.ema.load_state_dict(payload["ema_state"])
        self.global_step = int(payload["global_step"])

    def fit(self, metrics_path: Path | None = None,
            on_checkpoint: Callable[[Trainer], dict[str, Any]] | None = None) -> dict[str, Any]:
        """Train to ``max_steps``. Logs metrics (JSON lines) and calls ``on_checkpoint`` every
        ``ckpt_every`` steps; returns the timing and the collected checkpoint records."""
        t = self.cfg.train
        wall0, cpu0 = time.perf_counter(), time.process_time()
        window: list[dict[str, float]] = []
        checkpoints: list[dict[str, Any]] = []
        with contextlib.ExitStack() as stack:
            out = stack.enter_context(Path(metrics_path).open("a")) if metrics_path else None
            while self.global_step < t.max_steps:
                window.append(self.step())
                s = self.global_step
                rows = []
                if s % t.log_every == 0:
                    g = np.array([w["grad_norm"] for w in window])
                    rows.append({"kind": "train", "step": s, "epoch": self.sampler.epoch,
                                 "loss": float(np.mean([w["loss"] for w in window])),
                                 "grad_norm_mean": float(g.mean()),
                                 "grad_norm_max": float(g.max()), "lr": window[-1]["lr"],
                                 "wall_s": time.perf_counter() - wall0})
                    window = []
                if s % t.val_every == 0 or s % t.ckpt_every == 0:
                    net = self.inference_model
                    row = {"kind": "offline", "step": s,
                           "network": "ema" if self.ema is not None else "trained",
                           "train": offline_metrics(net, self.train_data, sample=False)}
                    if self.valid_data is not None:
                        row["valid"] = offline_metrics(net, self.valid_data)
                    rows.append(row)
                if s % t.ckpt_every == 0 and on_checkpoint is not None:
                    checkpoints.append({"step": s, **on_checkpoint(self)})
                if out is not None:
                    for r in rows:
                        out.write(json.dumps(r, allow_nan=False) + "\n")
                    out.flush()
        return {"wall_s": time.perf_counter() - wall0, "cpu_s": time.process_time() - cpu0,
                "checkpoints": checkpoints}
