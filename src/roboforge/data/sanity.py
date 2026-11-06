"""Per-episode integrity checks for robomimic low_dim demonstrations.

An episode that fails any check is *flagged* with its reasons. It is never deleted: the
split manifest lists flagged episodes, and the window dataset refuses to load them unless
the caller opts in explicitly. The checks cover array lengths, finiteness, action bounds,
the control-step spacing of saved simulator time, ``obs[t+1] == next_obs[t]`` alignment,
and the recorded task-success signal at the end of the episode.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

TIME_ATOL = 1e-6


def episode_issues(g: h5py.Group, control_dt: float, action_low: np.ndarray,
                   action_high: np.ndarray) -> list[str]:
    """Return a list of human-readable problems with one demo group (empty if clean)."""
    issues: list[str] = []
    n = int(g.attrs["num_samples"])
    arrays: dict[str, np.ndarray] = {"actions": g["actions"][()], "states": g["states"][()],
                                     "rewards": g["rewards"][()], "dones": g["dones"][()]}
    for kind in ("obs", "next_obs"):
        for k in g[kind]:
            arrays[f"{kind}/{k}"] = g[kind][k][()]
    bad_len = sorted(k for k, a in arrays.items() if len(a) != n)
    if bad_len:
        issues.append(f"length != num_samples ({n}): {bad_len}")
        return issues
    if n < 1:
        return ["empty episode"]
    nonfinite = sorted(k for k, a in arrays.items()
                       if np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all())
    if nonfinite:
        issues.append(f"non-finite values in {nonfinite}")
    act = arrays["actions"]
    if (act < action_low - 1e-9).any() or (act > action_high + 1e-9).any():
        issues.append("actions outside the action-spec bounds")
    dt = np.diff(arrays["states"][:, 0])
    if n > 1 and not np.allclose(dt, control_dt, atol=TIME_ATOL, rtol=0):
        issues.append(f"state time steps not {control_dt} s (range {dt.min()}..{dt.max()})")
    for k in g["obs"]:
        if n > 1 and not np.array_equal(arrays[f"obs/{k}"][1:], arrays[f"next_obs/{k}"][:-1]):
            issues.append(f"obs/{k}[t+1] != next_obs/{k}[t]")
    if float(arrays["rewards"][-1]) <= 0 or int(arrays["dones"][-1]) != 1:
        issues.append("final step has no success reward/done")
    return issues


def success_tail(g: h5py.Group) -> dict[str, Any]:
    """Describe where the success reward first appears and whether it then persists."""
    r = g["rewards"][()]
    pos = np.flatnonzero(r > 0)
    if len(pos) == 0:
        return {"first_success_step": None, "success_steps": 0, "persists_to_end": False}
    first = int(pos[0])
    return {"first_success_step": first, "success_steps": int(len(pos)),
            "steps_after_first_success": int(len(r) - 1 - first),
            "persists_to_end": bool(np.all(r[first:] > 0))}
