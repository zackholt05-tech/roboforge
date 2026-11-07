"""Planned uncertainty analysis (RESEARCH_PROTOCOL §9, §14.6; DECISIONS D-066).

**Hierarchical paired bootstrap.** Input: per family, an outcome matrix ``X[f]`` of shape
``(n_seeds_f, n_states)`` (success 0/1, or the censored episode length), with the same state
order for every family. Each of ``B`` replicates

1. resamples the ``n_states`` state indices with replacement, **one draw shared by every
   family** (pairing by ``state_id`` is kept), and
2. resamples each family's training seeds with replacement, **independently per family**;

the statistic is the mean of ``X[f][seeds, states]`` over the resampled seeds × states.
Paired differences are taken on the same replicates. Intervals are 95 % percentile intervals
(``numpy.percentile``, linear interpolation). Draw order, fixed: one generator
``numpy.random.default_rng(seed)``; first the ``(B, n_states)`` state indices, then, for
each family in the given order, its ``(B, n_seeds_f)`` seed indices.

Repeated stochastic samples at one (seed, state) would be averaged before calling this (R = 1
in Level 1, so there are none). Time steps are never units of analysis.

A difference is **separated** when its interval excludes 0 (§14.6) and **not separated**
otherwise; no p-values are computed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

B_DEFAULT = 10_000
BOOTSTRAP_SEED = 20261106
LEVEL = 0.95


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % by default)."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _interval(values: np.ndarray) -> list[float]:
    lo, hi = (100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2)
    return [float(np.percentile(values, lo)), float(np.percentile(values, hi))]


def separation(interval: list[float]) -> str:
    """``separated (+)`` / ``separated (-)`` when the interval excludes 0, else
    ``not separated``."""
    if interval[0] > 0:
        return "separated (+)"
    if interval[1] < 0:
        return "separated (-)"
    return "not separated"


def hierarchical_paired_bootstrap(outcomes: dict[str, np.ndarray],
                                  comparisons: list[tuple[str, str]], *,
                                  b: int = B_DEFAULT, seed: int = BOOTSTRAP_SEED,
                                  order: list[str] | None = None) -> dict[str, Any]:
    """Family means and paired differences ``a - b`` with 95 % percentile intervals.

    ``outcomes[f]`` is ``(n_seeds_f, n_states)``; every family has the same ``n_states``.
    ``order`` fixes the per-family draw order (default: insertion order of ``outcomes``)."""
    order = list(outcomes) if order is None else list(order)
    if sorted(order) != sorted(outcomes):
        raise ValueError("order must list every family exactly once")
    mats = {f: np.asarray(outcomes[f], dtype=np.float64) for f in order}
    n_states = {m.shape[1] for m in mats.values()}
    if len(n_states) != 1 or any(m.ndim != 2 or m.shape[0] < 1 for m in mats.values()):
        raise ValueError("every outcome matrix must be (n_seeds >= 1, n_states), same n_states")
    if any(not np.all(np.isfinite(m)) for m in mats.values()):
        raise ValueError("outcomes must be finite")
    (t,) = n_states
    rng = np.random.default_rng(seed)
    st = rng.integers(0, t, size=(b, t))
    reps = {}
    for f in order:
        m = mats[f]
        sd = rng.integers(0, m.shape[0], size=(b, m.shape[0]))
        per_seed = m[:, st].mean(axis=2)  # (n_seeds, B): each seed's mean over resampled states
        reps[f] = np.take_along_axis(per_seed.T, sd, axis=1).mean(axis=1)
    fam = {f: {"estimate": float(mats[f].mean()), "per_seed": mats[f].mean(axis=1).tolist(),
               "n_seeds": int(mats[f].shape[0]), "n_states": t,
               "ci95": _interval(reps[f])} for f in order}
    diffs = []
    for a, c in comparisons:
        d = reps[a] - reps[c]
        ci = _interval(d)
        diffs.append({"a": a, "b": c, "difference": float(mats[a].mean() - mats[c].mean()),
                      "ci95": ci, "separation": separation(ci)})
    return {"method": "hierarchical paired bootstrap (RESEARCH_PROTOCOL §9)", "B": b,
            "seed": seed, "level": LEVEL, "draw_order": ["states", *order],
            "interval": "percentile, numpy linear interpolation", "families": fam,
            "differences": diffs}
