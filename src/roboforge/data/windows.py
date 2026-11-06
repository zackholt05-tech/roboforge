"""Episode-bounded window index arithmetic shared by every sequence sampler.

The Session 0 loader that read raw robomimic ``obs`` keys was removed in Session 1: for Can
it would have fed the lagged relative-pose block (KI-003) to a policy. All model data now
flows through :class:`roboforge.data.sequence.SequenceDataset`, which uses these indices.

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

from dataclasses import dataclass

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
