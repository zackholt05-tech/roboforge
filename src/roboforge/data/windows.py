"""Episode-bounded (observation history, future action chunk) windows.

**Session 0 status: minimal feasibility loader.** It proves the data path from
HDF5 to a model batch; the audited pipeline (splits, train-only normalization,
manifests, hashing) is built in Session 1 on top of these index semantics.

Conventions for a decision at step ``t`` of an episode with ``T`` steps:

* ``obs_history[k] = obs[clip(t - T_o + 1 + k, 0, T-1)]`` for ``k = 0..T_o-1``
  (oldest first; before the episode start the first observation is repeated and
  ``obs_valid[k] = False``).
* ``future_action[k] = actions[t + k]`` for ``k = 0..T_p-1``; past the episode end
  the last action is repeated and ``action_valid[k] = False`` so losses can mask it.
* Windows never cross episode boundaries, and observations at indices ``> t`` are
  never read, so future states cannot leak into the policy input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class WindowIndex:
    """Index arrays for one window (into a single episode's arrays)."""

    obs_idx: np.ndarray
    obs_valid: np.ndarray
    act_idx: np.ndarray
    act_valid: np.ndarray


def window_indices(episode_len: int, t: int, obs_horizon: int, pred_horizon: int) -> WindowIndex:
    """Compute clipped indices and validity masks for decision step ``t``."""
    if episode_len <= 0:
        raise ValueError("episode_len must be positive")
    if not 0 <= t < episode_len:
        raise ValueError(f"t={t} outside episode of length {episode_len}")
    if obs_horizon < 1 or pred_horizon < 1:
        raise ValueError("horizons must be >= 1")
    raw_obs = np.arange(t - obs_horizon + 1, t + 1)
    raw_act = np.arange(t, t + pred_horizon)
    return WindowIndex(
        obs_idx=np.clip(raw_obs, 0, episode_len - 1),
        obs_valid=raw_obs >= 0,
        act_idx=np.clip(raw_act, 0, episode_len - 1),
        act_valid=raw_act < episode_len,
    )


class LowDimWindowDataset:
    """In-memory window dataset over selected demos of a robomimic low-dim HDF5 file.

    Each item is a dict of numpy arrays: ``obs`` ``[T_o, D_obs]`` (keys concatenated
    in the given order), ``obs_valid`` ``[T_o]``, ``action`` ``[T_p, A]``,
    ``action_valid`` ``[T_p]``, plus ``episode`` (demo name) and ``t``.
    """

    def __init__(self, path: Path, demos: Sequence[str], obs_keys: Sequence[str],
                 obs_horizon: int = 2, pred_horizon: int = 16) -> None:
        self.obs_keys = list(obs_keys)
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.episodes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        with h5py.File(path, "r") as f:
            for d in demos:
                g = f["data"][d]
                obs = np.concatenate([g["obs"][k][()].reshape(len(g["actions"]), -1)
                                      for k in self.obs_keys], axis=1).astype(np.float32)
                self.episodes[d] = (obs, g["actions"][()].astype(np.float32))
        self.index = [(d, t) for d, (_, a) in self.episodes.items() for t in range(len(a))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, object]:
        demo, t = self.index[i]
        obs, actions = self.episodes[demo]
        w = window_indices(len(actions), t, self.obs_horizon, self.pred_horizon)
        return {"obs": obs[w.obs_idx], "obs_valid": w.obs_valid, "action": actions[w.act_idx],
                "action_valid": w.act_valid, "episode": demo, "t": t}
