"""In-memory window tensors and the seeded batch sampler (RESEARCH_PROTOCOL §6.2).

The low-dim datasets are small (Lift 9,666 windows, Can 23,207), so every window of a split is
materialized once from :class:`roboforge.data.sequence.SequenceDataset`, which is the only
source of training data (D-018, D-022). Batches are then indexed tensors. There are no
loader workers and no hidden global RNG.

:class:`EpochStreamSampler` concatenates one seeded random permutation of all windows per
epoch and serves consecutive batches from that stream, so every window is seen equally
often. Its full state (generator state, current permutation, position) goes into every
checkpoint, which makes resumed training bitwise identical to uninterrupted training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from roboforge.data.sequence import SequenceDataset


@dataclass(frozen=True)
class WindowTensors:
    """All windows of one split, stacked."""

    obs_history: torch.Tensor  # [N, T_o, D] float32, normalized
    obs_valid: torch.Tensor  # [N, T_o] bool
    future_action: torch.Tensor  # [N, T_p, A] float32, normalized
    valid_mask: torch.Tensor  # [N, T_p] bool
    episode_id: tuple[str, ...]
    start_t: torch.Tensor  # [N] int64

    def __len__(self) -> int:
        return int(self.obs_history.shape[0])

    @classmethod
    def from_dataset(cls, ds: SequenceDataset) -> WindowTensors:
        """Materialize every window of ``ds`` (``SequenceDataset.__getitem__`` semantics)."""
        items = [ds[i] for i in range(len(ds))]
        if not items:
            raise ValueError("dataset has no windows")
        return cls(
            obs_history=torch.as_tensor(np.stack([it["obs_history"] for it in items])),
            obs_valid=torch.as_tensor(np.stack([it["obs_valid"] for it in items])),
            future_action=torch.as_tensor(np.stack([it["future_action"] for it in items])),
            valid_mask=torch.as_tensor(np.stack([it["valid_mask"] for it in items])),
            episode_id=tuple(it["episode_id"] for it in items),
            start_t=torch.as_tensor([it["start_t"] for it in items], dtype=torch.int64))

    def batch(self, idx: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
        """The windows at ``idx``."""
        i = torch.as_tensor(idx, dtype=torch.int64)
        return {"obs_history": self.obs_history[i], "obs_valid": self.obs_valid[i],
                "future_action": self.future_action[i], "valid_mask": self.valid_mask[i]}

    def subset(self, episodes: set[str]) -> WindowTensors:
        """Only the windows of the given episodes (for deliberate overfit tests)."""
        keep = [i for i, e in enumerate(self.episode_id) if e in episodes]
        if not keep:
            raise ValueError(f"no windows for episodes {sorted(episodes)}")
        i = torch.as_tensor(keep, dtype=torch.int64)
        return WindowTensors(self.obs_history[i], self.obs_valid[i], self.future_action[i],
                             self.valid_mask[i], tuple(self.episode_id[k] for k in keep),
                             self.start_t[i])


class EpochStreamSampler:
    """Consecutive batches from a stream of seeded per-epoch permutations."""

    def __init__(self, n: int, batch_size: int, seed: int) -> None:
        if n < 1 or batch_size < 1:
            raise ValueError("n and batch_size must be >= 1")
        self.n, self.batch_size = int(n), int(batch_size)
        self.rng = np.random.default_rng([int(seed), 1])
        self.perm = self.rng.permutation(self.n)
        self.pos, self.epoch = 0, 0

    def next_batch(self) -> np.ndarray:
        """The next ``batch_size`` window indices."""
        out: list[np.ndarray] = []
        need = self.batch_size
        while need:
            if self.pos == self.n:
                self.perm, self.pos = self.rng.permutation(self.n), 0
                self.epoch += 1
            take = min(need, self.n - self.pos)
            out.append(self.perm[self.pos:self.pos + take])
            self.pos += take
            need -= take
        return np.concatenate(out)

    def state_dict(self) -> dict[str, Any]:
        """Checkpointable state (tensors and JSON-compatible values only)."""
        return {"n": self.n, "batch_size": self.batch_size, "pos": self.pos,
                "epoch": self.epoch, "perm": torch.as_tensor(self.perm, dtype=torch.int64),
                "rng_json": json.dumps(self.rng.bit_generator.state)}

    def load_state_dict(self, s: dict[str, Any]) -> None:
        """Restore :meth:`state_dict` output (sizes must match)."""
        if (int(s["n"]), int(s["batch_size"])) != (self.n, self.batch_size):
            raise ValueError("sampler state is for a different dataset size or batch size")
        self.rng.bit_generator.state = json.loads(s["rng_json"])
        self.perm = s["perm"].numpy().astype(np.int64)
        self.pos, self.epoch = int(s["pos"]), int(s["epoch"])
