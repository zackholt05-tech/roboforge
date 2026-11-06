"""Losses shared by every policy family (RESEARCH_PROTOCOL §6.1).

Every family's training loss is a mean squared error in the normalized action space,
masked to the valid (non-padded) action steps of each window (D-022). For BC families the
error is between predicted and target actions; for diffusion it is between predicted and
true noise.
"""

from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid (non-padded) action steps and all action dimensions."""
    mask = valid.to(pred.dtype).unsqueeze(-1)
    denom = mask.sum() * target.shape[-1]
    if float(denom) == 0.0:
        raise ValueError("batch has no valid action steps")
    return (((pred - target) ** 2) * mask).sum() / denom
