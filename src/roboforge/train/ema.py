"""Exponential moving average of model weights (diffusion only; RESEARCH_PROTOCOL §6.1).

Re-implements the semantics of ``diffusers==0.11.1`` ``training_utils.EMAModel`` (Apache-2.0),
which robomimic ``d309eae`` Diffusion Policy uses with ``power=0.75``. After the ``k``-th call
of :meth:`EMA.step` (``k = 0, 1, ...``) every parameter is updated as
``ema = decay_k * ema + (1 - decay_k) * param`` with

    s = max(0, k - update_after_step - 1)
    decay_k = 0 if s == 0 else clamp(1 - (1 + s / inv_gamma) ** -power, min_value, max_value)

so the first two updates copy the weights. Buffers are copied. The state (averaged weights,
update count and last decay) is checkpointed, so resumed training stays bitwise identical;
``tests/unit/test_diffusion_reference.py`` checks it bitwise against diffusers.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class EMA:
    """EMA copy of a model, updated after every optimizer step."""

    def __init__(self, model: nn.Module, power: float, inv_gamma: float = 1.0,
                 min_value: float = 0.0, max_value: float = 0.9999,
                 update_after_step: int = 0) -> None:
        self.averaged_model = copy.deepcopy(model).eval()
        self.averaged_model.requires_grad_(False)
        self.power, self.inv_gamma = float(power), float(inv_gamma)
        self.min_value, self.max_value = float(min_value), float(max_value)
        self.update_after_step = int(update_after_step)
        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        """Decay factor used by the update with index ``optimization_step``."""
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        """Fold the current weights of ``model`` into the average."""
        self.decay = self.get_decay(self.optimization_step)
        ema = dict(self.averaged_model.named_parameters())
        for name, param in model.named_parameters():
            target = ema[name]
            if not param.requires_grad:
                target.copy_(param.data.to(dtype=target.dtype))
            else:
                target.mul_(self.decay)
                target.add_(param.data.to(dtype=target.dtype), alpha=1 - self.decay)
        ema_buffers = dict(self.averaged_model.named_buffers())
        for name, buf in model.named_buffers():
            ema_buffers[name].copy_(buf)
        self.optimization_step += 1

    def config(self) -> dict[str, Any]:
        """JSON-compatible hyperparameters."""
        return {"power": self.power, "inv_gamma": self.inv_gamma, "min_value": self.min_value,
                "max_value": self.max_value, "update_after_step": self.update_after_step,
                "reference": "diffusers 0.11.1 training_utils.EMAModel semantics"}

    def state_dict(self) -> dict[str, Any]:
        """Checkpointable state (tensors and JSON-compatible values only)."""
        return {"model_state": self.averaged_model.state_dict(),
                "optimization_step": self.optimization_step, "decay": self.decay,
                "config": self.config()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state_dict` output (strict on the weights and hyperparameters)."""
        if state["config"] != self.config():
            raise ValueError(f"EMA config {state['config']} != {self.config()}")
        self.averaged_model.load_state_dict(state["model_state"], strict=True)
        self.optimization_step = int(state["optimization_step"])
        self.decay = float(state["decay"])
