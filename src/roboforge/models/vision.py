"""Image encoders for the RGB observation-modality study (Session 8; candidates, not frozen).

Two trunks, each followed by robomimic's ``SpatialSoftmax`` pooling and a linear projection,
the ``VisualCore`` design of robomimic ``d309eae`` (``models/obs_core.py``,
``models/base_nets.py``; MIT) used by its image BC and Diffusion Policy configs:

* ``resnet18``: torchvision ResNet-18 convolutional trunk (``conv1`` … ``layer4``), randomly
  initialized (``weights=None``: nothing is downloaded), with every BatchNorm replaced by
  GroupNorm as in Diffusion Policy (Chi et al., RSS 2023), which keeps the network a pure
  function of its input under EMA and small batches. 11.2 M trunk parameters.
* ``compact``: a 4-stage residual trunk with widths (16, 32, 64, 128), GroupNorm and stride 2
  per stage, 0.32 M parameters per camera: a CPU-sized variant with the same pooling and output.

``SpatialSoftmax`` maps a ``C × h × w`` feature map to ``num_kp`` expected 2-D keypoints in
``[-1, 1]`` (Finn et al., "Deep Spatial Autoencoders for Visuomotor Learning", ICRA 2016),
reimplemented here with robomimic's conventions (1 × 1 conv to ``num_kp`` channels, fixed
temperature 1, ``linspace(-1, 1)`` grids, output ``[x_1, y_1, …]``).

Crops follow robomimic's ``CropRandomizer``: a random ``crop × crop`` window per image while
training and the centre window at evaluation. Inputs are ``uint8`` images ``[..., H, W, 3]``;
they are scaled to ``[0, 1]`` and channel-first inside the encoder.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

TRUNKS = ("resnet18", "compact")


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=max(1, channels // 16), num_channels=channels)


def resnet18_trunk() -> nn.Module:
    """torchvision ResNet-18 up to ``layer4``, BatchNorm → GroupNorm, random init."""
    from torchvision.models import resnet18

    net = resnet18(weights=None, norm_layer=_group_norm)
    return nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                         net.layer1, net.layer2, net.layer3, net.layer4)


class _Block(nn.Module):
    """Residual block: two 3 × 3 convs with GroupNorm; stride-2 projection shortcut."""

    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.n1 = _group_norm(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.n2 = _group_norm(c_out)
        self.skip = (nn.Identity() if stride == 1 and c_in == c_out else
                     nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                   _group_norm(c_out)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual update."""
        y = torch.relu(self.n1(self.conv1(x)))
        return torch.relu(self.n2(self.conv2(y)) + self.skip(x))


def compact_trunk(widths: Sequence[int] = (16, 32, 64, 128)) -> nn.Module:
    """Stem 3 × 3 conv, then one stride-2 residual block per width."""
    layers: list[nn.Module] = [nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                               _group_norm(widths[0]), nn.ReLU()]
    c = widths[0]
    for w in widths:
        layers.append(_Block(c, w, 2))
        c = w
    return nn.Sequential(*layers)


class SpatialSoftmax(nn.Module):
    """``[B, C, h, w]`` → ``[B, 2 · num_kp]`` expected keypoint coordinates in ``[-1, 1]``."""

    def __init__(self, channels: int, height: int, width: int, num_kp: int = 32) -> None:
        super().__init__()
        self.keypoints = nn.Conv2d(channels, num_kp, kernel_size=1)
        self.num_kp = num_kp
        ys, xs = torch.meshgrid(torch.linspace(-1.0, 1.0, height),
                                torch.linspace(-1.0, 1.0, width), indexing="ij")
        self.register_buffer("pos_x", xs.reshape(1, height * width))
        self.register_buffer("pos_y", ys.reshape(1, height * width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax over each keypoint map, then the expected x and y."""
        f = self.keypoints(x)
        b, k, h, w = f.shape
        attn = torch.softmax(f.reshape(b * k, h * w), dim=-1)
        ex = (attn * self.pos_x).sum(-1, keepdim=True)
        ey = (attn * self.pos_y).sum(-1, keepdim=True)
        return torch.cat([ex, ey], dim=-1).reshape(b, 2 * k)


def crop(images: torch.Tensor, size: int, generator: torch.Generator | None,
         train: bool) -> torch.Tensor:
    """``[N, H, W, C]`` → ``[N, size, size, C]``: random windows (train) or the centre window."""
    n, h, w, _ = images.shape
    if size > min(h, w):
        raise ValueError(f"crop {size} larger than image {h}x{w}")
    if not train:
        top, left = (h - size) // 2, (w - size) // 2
        return images[:, top:top + size, left:left + size]
    tops = torch.randint(0, h - size + 1, (n,), generator=generator)
    lefts = torch.randint(0, w - size + 1, (n,), generator=generator)
    rows = tops[:, None] + torch.arange(size)[None]
    cols = lefts[:, None] + torch.arange(size)[None]
    idx = torch.arange(n)[:, None, None]
    return images[idx, rows[:, :, None], cols[:, None, :]]


class VisualCore(nn.Module):
    """One camera: crop → [0, 1] channel-first → trunk → SpatialSoftmax → Linear(feature_dim)."""

    def __init__(self, trunk: str, image_size: int, crop_size: int, feature_dim: int = 64,
                 num_kp: int = 32) -> None:
        super().__init__()
        if trunk not in TRUNKS:
            raise ValueError(f"trunk must be one of {TRUNKS}")
        self.trunk_name, self.image_size, self.crop_size = trunk, image_size, crop_size
        self.trunk = resnet18_trunk() if trunk == "resnet18" else compact_trunk()
        with torch.no_grad():
            c, h, w = self.trunk(torch.zeros(1, 3, crop_size, crop_size)).shape[1:]
        self.pool = SpatialSoftmax(c, h, w, num_kp)
        self.proj = nn.Linear(2 * num_kp, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, images: torch.Tensor, *, train: bool = False,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """``uint8 [N, H, W, 3]`` → ``float32 [N, feature_dim]``."""
        x = crop(images, self.crop_size, generator, train)
        x = x.permute(0, 3, 1, 2).to(torch.float32).div(255.0)
        return self.proj(self.pool(self.trunk(x)))


class RGBObservationEncoder(nn.Module):
    """Per-camera ``VisualCore`` (separate weights, as robomimic) + proprioception, per frame.

    Input: ``images`` ``uint8 [B, T, n_cameras, H, W, 3]`` and normalized ``proprio`` ``[B, T, P]``.
    Output: ``[B, T, n_cameras · feature_dim + P]``. No object state enters this module.
    """

    def __init__(self, cameras: Sequence[str], proprio_dim: int, trunk: str, image_size: int,
                 crop_size: int, feature_dim: int = 64, num_kp: int = 32) -> None:
        super().__init__()
        self.cameras = tuple(cameras)
        self.cores = nn.ModuleList(VisualCore(trunk, image_size, crop_size, feature_dim, num_kp)
                                   for _ in self.cameras)
        self.proprio_dim = proprio_dim
        self.out_dim = len(self.cameras) * feature_dim + proprio_dim

    def forward(self, images: torch.Tensor, proprio: torch.Tensor, *, train: bool = False,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """Encode every frame of every sample."""
        b, t, n_cam = images.shape[:3]
        if n_cam != len(self.cameras) or proprio.shape[:2] != (b, t):
            raise ValueError(f"images {tuple(images.shape)} / proprio {tuple(proprio.shape)} "
                             f"do not match {len(self.cameras)} cameras")
        feats = [core(images[:, :, i].reshape(b * t, *images.shape[3:]), train=train,
                      generator=generator).reshape(b, t, -1)
                 for i, core in enumerate(self.cores)]
        return torch.cat([*feats, proprio.to(torch.float32)], dim=-1)


def parameter_count(module: nn.Module) -> int:
    """Trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
