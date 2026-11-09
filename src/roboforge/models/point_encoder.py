"""DP3-inspired point-cloud encoder for the Level 2 study (Session 9; MASTER_SPEC §7.3).

The design follows the "DP3 Encoder" of 3D Diffusion Policy (Ze, Zhang, Zhang, Hu, Wang and
Xu, "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D
Representations", RSS 2024, arXiv:2403.03954):
* a per-point MLP with three layers of widths 64, 128 and 256, each followed by LayerNorm and
  ReLU;
* a max pool over the points;
* a projection to a compact feature (Linear, then LayerNorm).

It is reimplemented here from that description. No DP3 code is copied. The result is
"DP3-inspired", not a reproduction of DP3:
* the clouds and training protocol are RoboForge's;
* the robot state enters as the raw normalized proprioception, exactly as in the RGB arm,
  instead of DP3's state MLP;
* the diffusion head is the RGB arm's.

**Input.** Points ``[..., N, 3]`` in the world frame, in metres. The fixed affine map of the
workspace box sends the box to ``[-1, 1]^3`` before the MLP. That map is a configuration
constant, not a learned or data-fitted normalization. Colour and segmentation are not used.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DP3Encoder(nn.Module):
    """``[B, N, 3]`` → ``[B, out_dim]``: per-point MLP, max pool, projection."""

    def __init__(self, widths: Sequence[int] = (64, 128, 256), out_dim: int = 64,
                 in_dim: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = in_dim
        for w in widths:
            layers += [nn.Linear(c, w), nn.LayerNorm(w), nn.ReLU()]
            c = w
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Linear(c, out_dim), nn.LayerNorm(out_dim))
        self.out_dim = out_dim

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Max over points of the per-point features, then the projection."""
        return self.proj(self.mlp(points).amax(dim=-2))


class PointCloudObservationEncoder(nn.Module):
    """One fused cloud + proprioception per frame (no object state enters this module).

    Input: ``points`` ``float32 [B, T, N, 3]`` (world metres) and normalized ``proprio``
    ``[B, T, P]``. Output: ``[B, T, out_dim + P]``."""

    def __init__(self, proprio_dim: int, box_centre: Sequence[float],
                 box_half_extent: Sequence[float], widths: Sequence[int] = (64, 128, 256),
                 out_dim: int = 64) -> None:
        super().__init__()
        self.core = DP3Encoder(widths, out_dim)
        self.register_buffer("box_centre", torch.tensor(list(box_centre), dtype=torch.float32))
        self.register_buffer("box_half", torch.tensor(list(box_half_extent), dtype=torch.float32))
        self.proprio_dim = proprio_dim
        self.out_dim = out_dim + proprio_dim

    def forward(self, points: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Encode every frame of every sample."""
        b, t, n, d = points.shape
        if d != 3 or proprio.shape[:2] != (b, t):
            raise ValueError(f"points {tuple(points.shape)} / proprio {tuple(proprio.shape)}")
        x = (points.to(torch.float32) - self.box_centre) / self.box_half
        f = self.core(x.reshape(b * t, n, 3)).reshape(b, t, -1)
        return torch.cat([f, proprio.to(torch.float32)], dim=-1)


__all__ = ["DP3Encoder", "PointCloudObservationEncoder"]
