"""Programmatic inspection of robomimic-format HDF5 demonstration files.

Produces a JSON-serializable summary (environment metadata, episode counts and
lengths, action statistics, observation keys/shapes, upstream filter masks) that
is written into the feasibility/data reports. Nothing here assumes a particular
action dimension or observation set: everything is read from the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def demo_keys(f: h5py.File) -> list[str]:
    """Return demonstration group names sorted by their integer suffix (``demo_0, demo_1, ...``)."""
    return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))


def env_args(f: h5py.File) -> dict[str, Any]:
    """Return the parsed ``data.attrs['env_args']`` JSON (env name, version, kwargs)."""
    return json.loads(f["data"].attrs["env_args"])


def _stats(x: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": x.min(axis=0).tolist(),
        "max": x.max(axis=0).tolist(),
        "mean": x.mean(axis=0).tolist(),
        "std": x.std(axis=0).tolist(),
    }


def summarize_hdf5(path: Path, max_demos_for_stats: int | None = None) -> dict[str, Any]:
    """Summarize a robomimic HDF5 file.

    Args:
        path: HDF5 file path.
        max_demos_for_stats: limit the number of episodes concatenated for action
            statistics (``None`` = all episodes).
    """
    with h5py.File(path, "r") as f:
        demos = demo_keys(f)
        if not demos:
            raise ValueError(f"{path}: 'data' group contains no demonstrations")
        first = f["data"][demos[0]]
        lengths = np.array([int(f["data"][d].attrs["num_samples"]) for d in demos])
        action_lengths = np.array([f["data"][d]["actions"].shape[0] for d in demos])
        stat_demos = demos if max_demos_for_stats is None else demos[:max_demos_for_stats]
        actions = np.concatenate([f["data"][d]["actions"][()] for d in stat_demos], axis=0)
        rewards_final = [
            float(f["data"][d]["rewards"][-1]) for d in demos if "rewards" in f["data"][d]
        ]
        dones_final = [int(f["data"][d]["dones"][-1]) for d in demos if "dones" in f["data"][d]]

        obs_shapes: dict[str, list[int]] = {}
        if "obs" in first:
            obs_shapes = {k: list(first["obs"][k].shape[1:]) for k in first["obs"]}
        masks = {}
        if "mask" in f:
            masks = {k: sorted(x.decode() for x in f["mask"][k][()]) for k in f["mask"]}

        summary: dict[str, Any] = {
            "path": str(path),
            "env_args": env_args(f),
            "total_samples_attr": int(f["data"].attrs.get("total", -1)),
            "n_demos": len(demos),
            "episode_length": {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mean": float(lengths.mean()),
                "median": float(np.median(lengths)),
                "sum": int(lengths.sum()),
            },
            "num_samples_matches_action_rows": bool(np.all(lengths == action_lengths)),
            "action_dim": int(actions.shape[1]),
            "action_stats": _stats(actions),
            "action_abs_le_1": bool(np.all(np.abs(actions) <= 1.0 + 1e-9)),
            "gripper_action_unique_values": sorted(set(np.round(actions[:, -1], 6).tolist()))[:10],
            "state_dim": int(first["states"].shape[1]) if "states" in first else None,
            "obs_keys": obs_shapes,
            "has_next_obs": "next_obs" in first,
            "has_model_file": "model_file" in first.attrs,
            "demo_group_keys": sorted(first.keys()),
            "final_reward_count_positive": int(sum(r > 0 for r in rewards_final)),
            "final_done_count": int(sum(dones_final)),
            "mask_sizes": {k: len(v) for k, v in masks.items()},
            "masks": masks,
        }
    return summary
