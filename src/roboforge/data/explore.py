"""Static dataset-explorer figures for the Session 1 audit (``reports/data_audit/*.png``).

Three figures per task, all drawn from the processed arrays and the committed split:

1. episode-length histogram, train vs valid (fraction of each split's demos);
2. per-dimension action distributions, train vs valid (small multiples, one shared y-axis
   per panel);
3. one example episode, chosen by a fixed rule (the lowest-numbered training demo): its
   feature blocks and actions over time.

These are static report images, so the hover layer that interactive charts need is replaced
by the full numeric tables in ``reports/data_audit.json``. Colours: categorical slots 1-4 of
the reference palette (the two-series train/valid pair was checked with the dataviz
validator: CVD ΔE 24.7, normal-vision ΔE 33.6, contrast ≥ 3:1 on the light surface).
Series identity is never carried by colour alone (legends plus panel titles).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2.0,
                         "legend.frameon": False})
    return plt


def length_histogram(lengths: dict[str, int], split: dict[str, Any], task: str,
                     out: Path) -> Path:
    """Grouped bars: fraction of each split's demos per 5-step length bin."""
    plt = _plt()
    lo = 5 * (min(lengths.values()) // 5)
    edges = np.arange(lo, max(lengths.values()) + 6, 5)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    width = 2.2
    for k, (name, ids) in enumerate(split["splits"].items()):
        counts = np.histogram([lengths[d] for d in ids], edges)[0]
        ax.bar(edges[:-1] + 0.2 + k * (width + 0.2), counts / len(ids), width=width,
               color=SERIES[k], label=f"{name} ({len(ids)} demos)", align="edge")
    ax.set_xticks(edges)
    ax.set_xlabel("episode length (control steps at 20 Hz), 5-step bins")
    ax.set_ylabel("fraction of the split's demos")
    ax.set_title(f"{task}: episode lengths by split", loc="left")
    ax.legend(loc="upper right")
    return _save(fig, out)


def action_distributions(episodes: dict[str, dict[str, np.ndarray]], split: dict[str, Any],
                         names: list[str], task: str, out: Path) -> Path:
    """Small multiples: density histogram of each action dimension, train vs valid."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(names), figsize=(2.0 * len(names), 2.6))
    bins = np.linspace(-1, 1, 41)
    for i, (ax, name) in enumerate(zip(axes, names, strict=True)):
        for k, (s, ids) in enumerate(split["splits"].items()):
            a = np.concatenate([episodes[d]["actions"][:, i] for d in ids])
            ax.hist(a, bins=bins, density=True, histtype="step", color=SERIES[k], label=s)
        ax.set_title(name, loc="left")
        ax.set_yticks([])
    axes[0].set_ylabel("density")
    axes[-1].legend(loc="upper left")
    fig.suptitle(f"{task}: dataset action distributions (raw units, bounds [-1, 1])",
                 x=0.01, ha="left")
    return _save(fig, out)


def example_episode(episodes: dict[str, dict[str, np.ndarray]], split: dict[str, Any],
                    slices: dict[str, tuple[int, int]], action_names: list[str], task: str,
                    out: Path) -> tuple[Path, str]:
    """Features and actions of the lowest-numbered training demo over time."""
    plt = _plt()
    demo = min(split["splits"]["train"], key=lambda d: int(d.split("_")[1]))
    obs, act = episodes[demo]["obs"], episodes[demo]["actions"]
    t = np.arange(len(obs)) * 0.05
    blocks = [*slices.items(), ("actions: dpos", (0, 3)), ("actions: drot", (3, 6)),
              ("actions: gripper", (6, 7))]
    fig, axes = plt.subplots(3, 3, figsize=(10, 7), sharex=True)
    for ax, (name, (lo, hi)) in zip(axes.flat, blocks, strict=False):
        src = act if name.startswith("actions") else obs
        labels = action_names[lo:hi] if src is act else _component_labels(name, hi - lo)
        for k, j in enumerate(range(lo, hi)):
            ax.plot(t, src[:, j], color=SERIES[k % len(SERIES)], label=labels[k])
        ax.set_title(name, loc="left")
        ax.legend(loc="best", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    for ax in list(axes.flat)[len(blocks):]:
        ax.set_visible(False)
    fig.suptitle(f"{task}: example episode {demo} (rule: lowest-numbered training demo); "
                 "features in raw units", x=0.01, ha="left")
    return _save(fig, out), demo


def _component_labels(name: str, n: int) -> list[str]:
    """Legend labels for the components of one feature block."""
    if name.endswith("_quat") and n == 4:
        return ["qx", "qy", "qz", "qw"]
    if name.endswith("gripper_qpos"):
        return [f"finger {j + 1}" for j in range(n)]
    return ["x", "y", "z"][:n] if n <= 3 else [f"[{j}]" for j in range(n)]


def _save(fig: Any, out: Path) -> Path:
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, metadata={"Software": None, "CreationDate": None})
    plt.close(fig)
    return out
