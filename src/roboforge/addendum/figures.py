"""Figures of the v1.1 addendum and the README chart, with a sidecar of every plotted value.

The style, palette and deterministic PDF writer are the paper's
(:mod:`roboforge.paper.figures`). The README chart is an SVG with a fixed hash salt, no date
and text drawn as paths, so it is byte-identical across rebuilds. Every builder returns
``(figure, data)``; ``data`` is exactly what is drawn, as percentages of the pinned fractions.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from roboforge.addendum.numbers import CONTRASTS  # noqa: E402
from roboforge.paper.figures import COLOR, RC, pdf_bytes  # noqa: E402

__all__ = ["FIGURES", "pdf_bytes", "readme_chart", "svg_bytes"]

STYLE = {"l1": ("o", "#6b6b6b"), "x3": ("s", "#0072B2"), "x2": ("D", "#CC79A7")}


def study_labels(src: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Legend text of each study, with seeds, starts and horizon read from the sources."""
    from roboforge.addendum.numbers import _x2_horizon

    x3, x2 = src["x3"], src["x2"]
    f = x3["frozen_families"]["diffusion_unet"]
    h = x2["primary"]["contrasts"][0]["frozen"]["horizon"]
    x2f = x2["primary"]["bootstrap"]["families"]["diffusion_unet"]
    return {
        "l1": f"Level 1 Can PH (frozen; {f['n_seeds']} seeds x {f['n_states']} starts, "
              f"{h} steps)",
        "x3": f"X3 separate replication ({len(x3['primary']['seeds'])} seeds x "
              f"{x3['suite']['n_states']} new starts, {h} steps)",
        "x2": f"X2 (Can MH; {x2f['n_seeds']} seeds x {x2f['n_states']} starts, "
              f"{_x2_horizon(x2)} steps; within-operator)",
        "level": f"{round(100 * x2['primary']['bootstrap']['level'])} %",
    }


CONTRAST_LABEL = {
    "h2bcmlp": "H2 diffusion - BC-MLP", "h2bcrnn": "H2 diffusion - BC-RNN",
    "h2c": "H2-c diffusion - BC-MLP-chunk", "h3": "H3 BC-RNN - BC-MLP",
}
X1_ARMS = (("diffusion_unet", "Diffusion U-Net", COLOR["diffusion_unet"], None),
           ("bc_mlp_chunk", "BC-MLP-chunk", COLOR["bc_mlp_chunk"], None),
           ("bc_mlp_chunk+clamp", "BC-MLP-chunk, clamped (X1)", COLOR["bc_mlp_chunk"], "//"))


def svg_bytes(fig: Figure) -> bytes:
    """Deterministic SVG bytes of a figure (no date, fixed hash salt, text as paths)."""
    buf = io.BytesIO()
    with plt.rc_context({"svg.hashsalt": "roboforge-v1.1", "svg.fonttype": "path"}):
        fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return buf.getvalue()


def _p(x: float) -> float:
    return round(100 * x, 4)


def _contrast_points(src: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    x3 = {c["name"]: c for c in src["x3"]["primary"]["contrasts"]}
    x2 = {c["name"]: c for c in src["x2"]["primary"]["contrasts"]}
    for k, name in CONTRASTS.items():
        f = x3[name]["frozen"]
        out[k] = {
            "l1": {"d": _p(f["difference"]), "ci": [_p(v) for v in f["bootstrap_ci95"]]},
            "x3": {"d": _p(x3[name]["difference"]),
                   "ci": [_p(v) for v in x3[name]["bootstrap_ci95"]]},
            "x2": {"d": _p(x2[name]["difference"]),
                   "ci": [_p(v) for v in x2[name]["bootstrap_ci95"]]},
        }
    return out


def _draw_contrasts(ax: Any, pts: dict[str, dict[str, Any]], level: str) -> None:
    keys = list(CONTRASTS)
    for i, k in enumerate(keys):
        for j, s in enumerate(("l1", "x3", "x2")):
            y = i + (j - 1) * 0.24
            v = pts[k][s]
            mk, col = STYLE[s]
            ax.errorbar(v["d"], y, xerr=[[v["d"] - v["ci"][0]], [v["ci"][1] - v["d"]]],
                        fmt=mk, color=col, ms=4, capsize=2, lw=1)
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_yticks(range(len(keys)), [CONTRAST_LABEL[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel(f"difference in success (pp), bootstrap {level} interval")


def contrasts(src: dict[str, dict[str, Any]]) -> tuple[Figure, dict[str, Any]]:
    """The four declared Can contrasts: frozen Level 1 Can PH, X3 and X2, side by side."""
    pts, lab = _contrast_points(src), study_labels(src)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.6, 2.8))
        _draw_contrasts(ax, pts, lab["level"])
        handles = [plt.Line2D([], [], marker=STYLE[k][0], color=STYLE[k][1], ls="",
                              label=lab[k]) for k in STYLE]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1,
                  frameon=False)
    return fig, {"unit": "pp", "contrasts": pts}


def _x1_points(src: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fams = src["x1"]["tasks"]["lift"]["bootstrap"]["families"]
    return {arm: {s: {"succ": _p(fams[f"{arm}:{s}"]["estimate"]),
                      "ci": [_p(v) for v in fams[f"{arm}:{s}"]["ci95"]]}
                  for s in ("robustness_base", "ood_arm_offset")} for arm, *_ in X1_ARMS}


def _draw_x1(ax: Any, pts: dict[str, dict[str, Any]], ylabel: str) -> None:
    sets = (("robustness_base", "base starts"), ("ood_arm_offset", "arm offset"))
    w = 0.26
    for i, (arm, label, col, hatch) in enumerate(X1_ARMS):
        for j, (s, _) in enumerate(sets):
            v = pts[arm][s]
            ax.bar(j + (i - 1) * w, v["succ"], w, color=col, hatch=hatch, edgecolor="white",
                   alpha=1.0 if hatch is None else 0.55, label=label if j == 0 else None)
            ax.errorbar(j + (i - 1) * w, v["succ"],
                        yerr=[[v["succ"] - v["ci"][0]], [v["ci"][1] - v["succ"]]],
                        color="#222222", lw=0.8, capsize=2)
    ax.set_xticks(range(len(sets)), [lab for _, lab in sets])
    ax.set_ylim(0, 105)
    ax.set_ylabel(ylabel)


def _x1_ylabel(src: dict[str, dict[str, Any]]) -> str:
    v = src["x1"]["tasks"]["lift"]["bootstrap"]["families"]["diffusion_unet:ood_arm_offset"]
    return f"Lift success (%), {v['n_seeds']} seeds x {v['n_states']} starts"


def x1_success(src: dict[str, dict[str, Any]]) -> tuple[Figure, dict[str, Any]]:
    """X1: Lift success on the base and arm-offset robustness sets, with and without the bound."""
    pts = _x1_points(src)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        _draw_x1(ax, pts, _x1_ylabel(src))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    return fig, {"unit": "%", "task": "lift", "arms": pts}


def readme_chart(src: dict[str, dict[str, Any]]) -> tuple[Figure, dict[str, Any]]:
    """The README chart: X1 on the left, the Can contrasts of X3 and X2 on the right."""
    x1, pts, lab = _x1_points(src), _contrast_points(src), study_labels(src)
    with plt.rc_context(RC):
        fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.2),
                                   gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.75})
        _draw_x1(a, x1, _x1_ylabel(src))
        a.set_title("X1, Lift (exploratory, after the freeze; validation regime)")
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=6.5)
        _draw_contrasts(b, pts, lab["level"])
        b.set_title("Can contrasts: frozen, X3 and X2 beside each other, never pooled")
        handles = [plt.Line2D([], [], marker=STYLE[k][0], color=STYLE[k][1], ls="",
                              label=lab[k]) for k in STYLE]
        b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=1,
                 frameon=False, fontsize=6.5)
    return fig, {"x1": {"unit": "%", "task": "lift", "arms": x1},
                 "contrasts": {"unit": "pp", "values": pts}}


FIGURES = {"contrasts": contrasts, "x1_success": x1_success}
