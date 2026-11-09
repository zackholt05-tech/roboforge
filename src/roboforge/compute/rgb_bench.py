"""Timing core for candidate RGB policies on any torch device (Session 8, Gate D).

**Timing only, not training.** The policy has random weights and the inputs are random:
``uint8`` images and normalized proprioception and actions. No dataset or evaluation state
is read, and no checkpoint is written. The output is seconds, memory and parameter hashes.

Candidate policy: :class:`~roboforge.models.vision.RGBObservationEncoder` (two cameras; per
camera a trunk, SpatialSoftmax(32) and Linear(64); proprioception 9-D) for each frame. Its
``T_o × 137`` window feeds one Level 1 head with its committed architecture config:

* ``bc_chunk``: the BC-MLP-chunk head of ``configs/models/bc_mlp_chunk.yaml``;
* ``diffusion``: the U-Net head of ``configs/models/diffusion_unet.yaml``, with ε-MSE at one
  sampled timestep.

One timed step mirrors :meth:`roboforge.train.loop.Trainer.step`:

1. copy the ``uint8`` batch from host memory to the device (pinned on CUDA);
2. forward with random crops;
3. loss (masked MSE, or the diffusion loss);
4. backward;
5. total gradient norm (``grad_clip`` is null in both configs);
6. AdamW step with the config's optimizer settings;
7. the EMA update when the head config enables it (diffusion: ``ema: true``).

Every row records this step definition (``step_def``). The Session 8 CPU numbers at
``f3c9170`` used an older step with no EMA and no gradient norm.

Device rules (KNOWN_ISSUES KI-030):

* one generator lives on the data's device and serves both the crops and the diffusion noise;
* the EMA copy is built after the model has moved.

Inference is never timed here: the frozen ``DiffusionPolicy.sample`` draws its noise on the
CPU, and every rollout runs on the canonical CPU container anyway.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from roboforge.models.losses import masked_mse
from roboforge.models.registry import build_model
from roboforge.models.vision import RGBObservationEncoder, parameter_count
from roboforge.paths import REPO_ROOT
from roboforge.train.ema import EMA

CAMERAS = ("agentview", "robot0_eye_in_hand")
PROPRIO_DIM, ACTION_DIM, T_O, T_P = 9, 7, 2, 16
CROP = {84: 76, 128: 116}
HEAD_CONFIG = {"bc_chunk": ("bc_mlp", "bc_mlp_chunk.yaml"),
               "diffusion": ("diffusion_unet", "diffusion_unet.yaml")}
MODES = ("default", "deterministic")
CUBLAS_WORKSPACE = ":4096:8"


def head_config(head: str) -> dict[str, Any]:
    """The committed Level 1 config of ``head``."""
    return yaml.safe_load((REPO_ROOT / "configs" / "models" / HEAD_CONFIG[head][1]).read_text())


class Policy(nn.Module):
    """Encoder + head as one module (what the optimizer and the EMA see)."""

    def __init__(self, trunk: str, size: int, head: str) -> None:
        super().__init__()
        cfg = head_config(head)
        self.head_name = head
        self.enc = RGBObservationEncoder(CAMERAS, PROPRIO_DIM, trunk, size, CROP[size])
        self.net = build_model(HEAD_CONFIG[head][0], cfg["model"], self.enc.out_dim,
                               ACTION_DIM, T_O, T_P)

    def loss(self, b: dict[str, torch.Tensor], gen: torch.Generator) -> torch.Tensor:
        """Training loss on one batch, with random crops and (diffusion) noise from ``gen``."""
        feats = self.enc(b["images"], b["proprio"], train=True, generator=gen)
        if self.head_name == "diffusion":
            return self.net.training_loss(feats, b["valid"], b["action"], b["mask"], gen)
        return masked_mse(self.net(feats, b["valid"]), b["action"], b["mask"])


def host_batch(b: int, size: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random inputs of the training shapes in host memory (fixed by ``seed``)."""
    g = torch.Generator().manual_seed(seed)
    return {"images": torch.randint(0, 256, (b, T_O, len(CAMERAS), size, size, 3),
                                    dtype=torch.uint8, generator=g),
            "proprio": torch.rand(b, T_O, PROPRIO_DIM, generator=g) * 2 - 1,
            "valid": torch.ones(b, T_O, dtype=torch.bool),
            "action": torch.rand(b, T_P, ACTION_DIM, generator=g) * 2 - 1,
            "mask": torch.ones(b, T_P, dtype=torch.bool)}


def set_mode(mode: str) -> dict[str, Any]:
    """Apply ``default`` (torch defaults) or ``deterministic`` and return the flags in force.

    ``deterministic`` means ``torch.use_deterministic_algorithms(True)``, cuDNN deterministic,
    cuDNN benchmark off. cuBLAS also needs ``CUBLAS_WORKSPACE_CONFIG`` set before its first
    call; :func:`roboforge.compute.rgb_probe.rgb_probe_job` sets it at job start."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    det = mode == "deterministic"
    torch.use_deterministic_algorithms(det)
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = False
    return {"mode": mode, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


class Bench:
    """One candidate on one device: model, optimizer, EMA, generator and a fixed host batch."""

    def __init__(self, trunk: str, size: int, head: str, batch: int, device: str,
                 seed: int = 0) -> None:
        self.dev = torch.device(device)
        cfg = head_config(head)
        torch.manual_seed(seed)
        self.policy = Policy(trunk, size, head).to(self.dev)
        o = cfg["optim"]
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=o["lr"],
                                     betas=tuple(o["betas"]), weight_decay=o["weight_decay"])
        m = cfg["model"]
        self.ema = (EMA(self.policy, power=m["ema_power"], max_value=m["ema_max_decay"])
                    if m.get("ema") else None)  # built after .to(device)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.host = host_batch(batch, size, seed)
        if self.dev.type == "cuda":
            self.host = {k: v.pin_memory() for k, v in self.host.items()}
        self.step_def = {"batch": batch, "h2d_copy_in_step": self.dev.type != "cpu",
                         "grad_norm": True, "grad_clip": cfg["train"]["grad_clip"],
                         "ema": self.ema is not None, "optimizer": "adamw", **o}
        self.meta = {"trunk": trunk, "size": size, "crop": CROP[size], "head": head,
                     "device": self.dev.type,
                     "device_name": (torch.cuda.get_device_name(self.dev)
                                     if self.dev.type == "cuda" else None),
                     "params_encoder": parameter_count(self.policy.enc),
                     "params_head": parameter_count(self.policy.net)}

    def step(self) -> float:
        """One optimizer step (see the module docstring); returns the loss."""
        b = {k: v.to(self.dev, non_blocking=True) for k, v in self.host.items()}
        self.policy.train()
        loss = self.policy.loss(b, self.gen)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.get_total_norm([p.grad for p in self.policy.parameters()])
        self.opt.step()
        if self.ema is not None:
            self.ema.step(self.policy)
        return float(loss.detach())

    def digest(self) -> str:
        """SHA-256 of every parameter and EMA parameter, as float32 bytes on the host."""
        h = hashlib.sha256()
        mods = [self.policy] + ([self.ema.averaged_model] if self.ema is not None else [])
        for mod in mods:
            for name, p in mod.state_dict().items():
                h.update(name.encode())
                h.update(p.detach().to("cpu").contiguous().numpy().tobytes())
        return h.hexdigest()


def train_steps(trunk: str, size: int, head: str, batch: int, steps: int, warmup: int,
                device: str = "cpu", mode: str = "default") -> dict[str, Any]:
    """Seconds per optimizer step after ``warmup`` steps, plus peak device memory."""
    flags = set_mode(mode)
    bench = Bench(trunk, size, head, batch, device)
    if bench.dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bench.dev)
    for _ in range(warmup):
        bench.step()
    _sync(bench.dev)
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        bench.step()
        _sync(bench.dev)
        times.append(time.perf_counter() - t0)
    t = np.array(times)
    peak = (torch.cuda.max_memory_allocated(bench.dev) / 2**20
            if bench.dev.type == "cuda" else None)
    return {**bench.meta, "flags": flags, "step_def": bench.step_def, "warmup": warmup,
            "steps_timed": steps, "median_s_per_step": float(np.median(t)),
            "mean_s_per_step": float(t.mean()), "max_s_per_step": float(t.max()),
            "steps_per_s_median": float(1 / np.median(t)), "peak_memory_mib": peak}


def repeatability(trunk: str, size: int, head: str, batch: int, steps: int,
                  device: str = "cpu", mode: str = "default") -> dict[str, Any]:
    """Run the same ``steps`` twice from the same seed; are the final parameters bitwise equal?"""
    flags = set_mode(mode)
    digests, losses = [], []
    for _ in range(2):
        bench = Bench(trunk, size, head, batch, device)
        losses.append([bench.step() for _ in range(steps)])
        _sync(bench.dev)
        digests.append(bench.digest())
    return {**bench.meta, "flags": flags, "step_def": bench.step_def, "steps": steps,
            "param_sha256": digests, "bitwise_equal": digests[0] == digests[1],
            "losses_equal": losses[0] == losses[1], "final_losses": [x[-1] for x in losses]}


__all__ = ["CAMERAS", "CROP", "CUBLAS_WORKSPACE", "HEAD_CONFIG", "MODES", "Bench", "Policy",
           "head_config", "host_batch", "repeatability", "set_mode", "train_steps"]
