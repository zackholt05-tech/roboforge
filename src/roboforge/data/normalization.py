"""Train-only, reversible per-dimension affine normalization (DECISIONS D-020).

``normalized = x * scale + offset`` and ``x = (normalized - offset) / scale``.

Modes:

* ``minmax`` maps the training range of each dimension to ``[-1, 1]``. Dimensions whose
  training range is below ``range_eps`` get ``scale = 1`` and are centred at 0. These are the
  semantics of Diffusion Policy's ``limits`` normalizer (Chi et al., 2023; MIT license),
  re-implemented here.
* ``zscore`` maps the training mean to 0 and standard deviation to 1. The standard deviation
  is floored at ``range_eps``.

Statistics are always computed from per-step arrays of **training episodes only**. They are
never computed from windows, whose padding would duplicate edge steps. The full statistics
are saved with the transform, so the normalizer can be audited and rebuilt, and are hashed
for provenance. :class:`ActionCodec` turns model outputs into simulator actions: it
inverts the transform and clips to the dataset's action bounds, counting clipped
dimensions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from roboforge.provenance import sha256_json

MODES = ("minmax", "zscore")


@dataclass(frozen=True)
class Normalizer:
    """Per-dimension affine transform plus the training statistics it was fit on."""

    mode: str
    range_eps: float
    scale: np.ndarray
    offset: np.ndarray
    stats: dict[str, Any]

    @classmethod
    def fit(cls, arrays: Sequence[np.ndarray], mode: str, range_eps: float = 1e-4) -> Normalizer:
        """Fit on a list of ``[T_i, D]`` per-episode arrays (training episodes only)."""
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        if not arrays:
            raise ValueError("cannot fit a normalizer on zero episodes")
        x = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("normalizer input must be a finite [N, D] array")
        lo, hi, mean, std = x.min(0), x.max(0), x.mean(0), x.std(0)
        if mode == "minmax":
            rng = hi - lo
            ignore = rng < range_eps
            scale = np.where(ignore, 1.0, 2.0 / np.where(ignore, 1.0, rng))
            offset = np.where(ignore, -lo, -1.0 - scale * lo)
        else:
            scale = 1.0 / np.maximum(std, range_eps)
            offset = -mean * scale
        stats = {"n_samples": int(x.shape[0]), "n_episodes": len(arrays), "dim": int(x.shape[1]),
                 "min": lo.tolist(), "max": hi.tolist(), "mean": mean.tolist(),
                 "std": std.tolist()}
        return cls(mode, float(range_eps), scale, offset, stats)

    @property
    def dim(self) -> int:
        """Number of dimensions."""
        return int(self.scale.shape[0])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply the transform (float64 arithmetic); broadcasts over leading dimensions."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dim {x.shape[-1]} != normalizer dim {self.dim}")
        return x * self.scale + self.offset

    def unnormalize(self, y: np.ndarray) -> np.ndarray:
        """Invert :meth:`normalize`."""
        y = np.asarray(y, dtype=np.float64)
        if y.shape[-1] != self.dim:
            raise ValueError(f"last dim {y.shape[-1]} != normalizer dim {self.dim}")
        return (y - self.offset) / self.scale

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form (exact float64 values via ``repr``)."""
        return {"mode": self.mode, "range_eps": self.range_eps, "scale": self.scale.tolist(),
                "offset": self.offset.tolist(), "stats": self.stats}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Normalizer:
        """Rebuild from :meth:`to_dict` output."""
        if d["mode"] not in MODES:
            raise ValueError(f"mode {d['mode']!r} not in {MODES}")
        scale, offset = np.asarray(d["scale"], np.float64), np.asarray(d["offset"], np.float64)
        if scale.shape != offset.shape or not np.all(scale > 0):
            raise ValueError("invalid normalizer: scale must be positive and match offset")
        return cls(d["mode"], float(d["range_eps"]), scale, offset, d["stats"])

    def sha256(self) -> str:
        """Hash of the canonical JSON form."""
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class ActionCodec:
    """Map between dataset actions, normalized model actions and clipped simulator actions."""

    normalizer: Normalizer
    low: np.ndarray
    high: np.ndarray

    def encode(self, dataset_action: np.ndarray) -> np.ndarray:
        """``dataset_action -> normalized_model_action`` (loss / diffusion target space)."""
        return self.normalizer.normalize(dataset_action)

    def decode(self, model_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``normalized_model_action -> clipped_env_action``; also returns the clip mask."""
        raw = self.normalizer.unnormalize(model_action)
        clipped = np.clip(raw, self.low, self.high)
        return clipped, clipped != raw
