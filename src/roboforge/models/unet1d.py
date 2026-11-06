"""Conditional 1D temporal U-Net noise-prediction network (MASTER_SPEC §5.5).

**Adapted third-party code.** The modules below are adapted from
``robomimic/models/diffusion_policy_nets.py`` at robomimic commit ``d309eae`` (MIT License,
Copyright (c) 2021 Stanford Vision and Learning Lab; license text in
``THIRD_PARTY_NOTICES.md``). That file is robomimic's copy of the ``ConditionalUnet1D`` of
Diffusion Policy (Cheng Chi et al., RSS 2023, https://github.com/real-stanford/diffusion_policy,
MIT License).

RoboForge changes: type hints, docstrings, an explicit horizon check, no ``print`` at
construction, and no implicit int-to-tensor timestep conversion. Parameter names and the
forward arithmetic are kept identical, so a robomimic ``ConditionalUnet1D`` state dict loads
into this module and produces bitwise-equal outputs
(``tests/unit/test_diffusion_reference.py``).

Structure, for down widths ``(d_1, ..., d_L)``:

* the diffusion timestep ``t`` is encoded by a sinusoidal embedding (``step_embed_dim``),
  then ``Linear -> Mish -> Linear``; it is concatenated with the global observation
  condition ``c`` into the FiLM condition;
* every ``ConditionalResidualBlock1D`` is ``Conv1d -> GroupNorm -> Mish`` twice, with a
  per-channel FiLM scale and bias predicted from the condition after the first conv, plus a
  1x1 residual projection;
* the encoder has ``L`` levels of two residual blocks; each level but the last halves the
  temporal length (stride-2 conv). Two residual blocks form the bottleneck. The decoder has
  ``L - 1`` levels that concatenate the matching skip, apply two residual blocks and double
  the length (transposed conv). A final ``Conv1dBlock`` and a 1x1 conv map back to the action
  dimension. The prediction horizon must therefore be divisible by ``2^(L-1)``.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of the (integer) diffusion timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B]`` timesteps -> ``[B, dim]`` (sin half, then cos half)."""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    """Halve the temporal length with a stride-2 convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, T] -> [B, C, T/2]``."""
        return self.conv(x)


class Upsample1d(nn.Module):
    """Double the temporal length with a transposed convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, T] -> [B, C, 2T]``."""
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """``Conv1d -> GroupNorm -> Mish``."""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int,
                 n_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C_in, T] -> [B, C_out, T]``."""
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two conv blocks with FiLM conditioning (per-channel scale and bias) and a residual."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 kernel_size: int = 3, n_groups: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2),
                                          nn.Unflatten(-1, (-1, 1)))
        self.residual_conv = (nn.Conv1d(in_channels, out_channels, 1)
                              if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x [B, C_in, T]``, ``cond [B, cond_dim]`` -> ``[B, C_out, T]``."""
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
        out = embed[:, 0, ...] * out + embed[:, 1, ...]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """Noise-prediction U-Net over the time axis of an action chunk."""

    def __init__(self, input_dim: int, global_cond_dim: int, diffusion_step_embed_dim: int,
                 down_dims: tuple[int, ...], kernel_size: int = 5, n_groups: int = 8) -> None:
        super().__init__()
        if len(down_dims) < 1:
            raise ValueError("down_dims needs at least one level")
        all_dims = [input_dim, *down_dims]
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(),
            nn.Linear(dsed * 4, dsed))
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:], strict=True))
        mid_dim = all_dims[-1]

        def block(i: int, o: int) -> ConditionalResidualBlock1D:
            return ConditionalResidualBlock1D(i, o, cond_dim=cond_dim, kernel_size=kernel_size,
                                              n_groups=n_groups)

        self.mid_modules = nn.ModuleList([block(mid_dim, mid_dim), block(mid_dim, mid_dim)])
        self.down_modules = nn.ModuleList([
            nn.ModuleList([block(d_in, d_out), block(d_out, d_out),
                           Downsample1d(d_out) if ind < len(in_out) - 1 else nn.Identity()])
            for ind, (d_in, d_out) in enumerate(in_out)])
        # As in the reference, every decoder level upsamples (its ``is_last`` test can never
        # be true there), which restores the full horizon after the L-1 encoder downsamples.
        self.up_modules = nn.ModuleList([
            nn.ModuleList([block(d_out * 2, d_in), block(d_in, d_in), Upsample1d(d_in)])
            for d_in, d_out in reversed(in_out[1:])])
        self.final_conv = nn.Sequential(Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
                                        nn.Conv1d(start_dim, input_dim, 1))
        self.n_levels = len(down_dims)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor,
                global_cond: torch.Tensor) -> torch.Tensor:
        """``sample [B, T, A]``, ``timestep [B]`` or scalar (int64), ``global_cond [B, G]``
        -> predicted noise ``[B, T, A]``."""
        if sample.shape[1] % (2 ** (self.n_levels - 1)):
            raise ValueError(f"horizon {sample.shape[1]} not divisible by "
                             f"2^{self.n_levels - 1}")
        x = sample.moveaxis(-1, -2)
        if timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timesteps = timestep.expand(sample.shape[0])
        global_feature = torch.cat([self.diffusion_step_encoder(timesteps), global_cond], axis=-1)
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet2(resnet(x, global_feature), global_feature)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = upsample(resnet2(resnet(x, global_feature), global_feature))
        return self.final_conv(x).moveaxis(-1, -2)
