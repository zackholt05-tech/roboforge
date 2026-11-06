"""Window sampler over the processed dataset: (observation history, future action chunk).

One item per decision step ``t`` of every episode in the requested split. The index semantics
are those of :func:`roboforge.data.windows.window_indices`:

* ``obs_history[k] = obs[clip(t - T_o + 1 + k, 0, T-1)]``, oldest first. Before the episode
  start the first observation is repeated and ``obs_valid[k] = False``.
* ``future_action[k] = actions[t + k]``. Past the episode end the last action is repeated and
  ``valid_mask[k] = False``, so losses must ignore it.
* ``start_t = t``: the step whose observation is the newest input and whose action is
  ``future_action[0]``. Observations at indices ``> t`` are never read.

Observations and actions are normalized with the committed train-only normalizers and
returned as float32. Every model family (BC-MLP, BC-RNN, diffusion) reads its data through
this class, so all of them see the same observation spec, split and normalization (D-018).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from roboforge.data.normalization import Normalizer
from roboforge.data.windows import window_indices


class SequenceDataset:
    """Episode-bounded windows over pre-loaded, verified episode arrays."""

    def __init__(self, episodes: dict[str, dict[str, np.ndarray]], demos: Sequence[str],
                 obs_normalizer: Normalizer, action_normalizer: Normalizer,
                 obs_horizon: int, pred_horizon: int) -> None:
        if obs_horizon < 1 or pred_horizon < 1:
            raise ValueError("horizons must be >= 1")
        missing = [d for d in demos if d not in episodes]
        if missing:
            raise KeyError(f"demos not loaded: {missing}")
        self.demos = list(demos)
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.obs_normalizer, self.action_normalizer = obs_normalizer, action_normalizer
        self._obs: dict[str, np.ndarray] = {}
        self._act: dict[str, np.ndarray] = {}
        for d in self.demos:
            obs, act = episodes[d]["obs"], episodes[d]["actions"]
            if len(obs) != len(act) or len(obs) == 0:
                raise ValueError(f"{d}: {len(obs)} observations vs {len(act)} actions")
            self._obs[d] = obs_normalizer.normalize(obs).astype(np.float32)
            self._act[d] = action_normalizer.normalize(act).astype(np.float32)
        self.index: list[tuple[str, int]] = [(d, t) for d in self.demos
                                             for t in range(len(self._act[d]))]

    @classmethod
    def from_committed(cls, task: str, split: str, *, allow_flagged: bool = False,
                       obs_horizon: int | None = None,
                       pred_horizon: int | None = None) -> SequenceDataset:
        """Build from the committed task artifacts and the hash-verified processed file.

        Raises if the processed file's content hash differs from the task manifest, or if
        the split contains flagged episodes and ``allow_flagged`` is false.
        """
        from roboforge.data.pipeline import load_task_artifacts
        from roboforge.data.processed import load_processed, processed_path

        docs = load_task_artifacts(task)
        m, sp = docs["task_manifest"], docs["split"]
        demos = sp["splits"][split]
        flagged = sorted(set(demos) & set(sp["flagged"]))
        if flagged and not allow_flagged:
            raise ValueError(f"{task}/{split}: flagged episodes {flagged}; see the split manifest")
        proc = m["processed"]
        episodes = load_processed(processed_path(task, proc["converter_version"]),
                                  expected_sha256=proc["content_sha256"])
        w = m["windows"]
        return cls(episodes, demos, Normalizer.from_dict(docs["normalization"]["obs"]),
                   Normalizer.from_dict(docs["normalization"]["action"]),
                   obs_horizon or int(w["obs_horizon"]), pred_horizon or int(w["pred_horizon"]))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        demo, t = self.index[i]
        obs, act = self._obs[demo], self._act[demo]
        w = window_indices(len(act), t, self.obs_horizon, self.pred_horizon)
        return {"obs_history": obs[w.obs_idx], "obs_valid": w.obs_valid,
                "future_action": act[w.act_idx], "valid_mask": w.act_valid,
                "episode_id": demo, "start_t": t,
                "obs_index": w.obs_idx, "action_index": w.act_idx}


def collate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack window items into torch tensors (``episode_id`` stays a list of strings)."""
    import torch

    out: dict[str, Any] = {"episode_id": [it["episode_id"] for it in items]}
    for k in ("obs_history", "obs_valid", "future_action", "valid_mask", "obs_index",
              "action_index"):
        out[k] = torch.as_tensor(np.stack([it[k] for it in items]))
    out["start_t"] = torch.as_tensor([it["start_t"] for it in items])
    return out
