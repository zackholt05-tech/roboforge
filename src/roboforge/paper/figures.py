"""Paper figures from the frozen sources, with a JSON sidecar of every plotted value (D-094).

matplotlib (locked through robomimic, DECISIONS D-093) with the Agg backend, embedded
TrueType fonts and no creation date, so the PDFs are byte-identical across rebuilds. Each
builder returns ``(figure, data)``; ``data`` is what the figure draws, written beside it.
Genuine simulator renders are copied unchanged, located by their pinned SHA-256.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from roboforge.paper import fmt  # noqa: E402
from roboforge.paper.numbers import L1_FAMILIES, LEARNED, TASKS  # noqa: E402
from roboforge.paper.sources import Sources, sha256_file  # noqa: E402

# Okabe-Ito colour-blind-safe palette; one colour per family across every figure.
COLOR = {
    "bc_mlp": "#E69F00", "bc_rnn": "#56B4E9", "bc_mlp_chunk": "#009E73",
    "diffusion_unet": "#D55E00", "random_uniform": "#999999",
    "rgb_bc_chunk": "#0072B2", "rgb_diffusion": "#CC79A7", "pc_diffusion": "#000000",
}
MARKER = {"lift": "o", "can": "s"}

RC = {
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    "svg.hashsalt": "roboforge", "path.simplify": False,
}


def pdf_bytes(fig: Figure) -> bytes:
    """Deterministic PDF bytes of a figure (no creation date)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    return buf.getvalue()


def _p(x: float) -> float:
    return round(100 * x, 4)


def _err(v: dict[str, Any]) -> tuple[float, list[float]]:
    est = 100 * v["estimate"]
    return est, [est - 100 * v["ci95"][0], 100 * v["ci95"][1] - est]


def heldout_success(src: Sources) -> tuple[Figure, dict[str, Any]]:
    """Level 1 held-out success per family with 95 % interval and per-seed values."""
    r = src.json("l1_results")
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
        data: dict[str, Any] = {}
        for ax, t in zip(axes, TASKS, strict=True):
            fams = r["tasks"][t]["success"]["families"]
            data[t] = {}
            for i, f in enumerate(L1_FAMILIES):
                v = fams[f]
                est, e = _err(v)
                ax.errorbar([i - 0.1], [est], yerr=[[e[0]], [e[1]]], fmt="o", color=COLOR[f],
                            capsize=3, ms=5, lw=1.2, zorder=3)
                xs = [i + d for d in (0.12, 0.22, 0.32)][: len(v["per_seed"])]
                ax.scatter(xs, [100 * s for s in v["per_seed"]], s=10, facecolors="white",
                           edgecolors=COLOR[f], lw=0.8, zorder=4)
                data[t][f] = {"estimate": _p(v["estimate"]), "ci95": [_p(x) for x in v["ci95"]],
                              "per_seed": [_p(x) for x in v["per_seed"]]}
            ax.set_xticks(range(len(L1_FAMILIES)))
            ax.set_xticklabels([fmt.FAMILY_LABEL[f].replace(" (uniform)", "")
                                .replace("Diffusion U-Net", "Diffusion") for f in L1_FAMILIES],
                               rotation=25, ha="right")
            n_seeds = fams["diffusion_unet"]["n_seeds"]
            ax.set_title(f"{t.capitalize()} held-out ({r['tasks'][t]['n_states']} starts "
                         f"$\\times$ {n_seeds} seeds)")
            ax.set_ylim(-5, 105)
            ax.grid(axis="y", lw=0.3, alpha=0.5)
        axes[0].set_ylabel("success (%)")
    return fig, data


def differences(src: Sources) -> tuple[Figure, dict[str, Any]]:
    """Declared paired differences in success (H2, H2-c, H3) per task, 95 % intervals."""
    r = src.json("l1_results")
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(4.6, 2.3))
        data: dict[str, Any] = {}
        labels = []
        rows = [d for d in r["tasks"]["lift"]["success"]["differences"] if d["hypothesis"] != "H1"]
        for j, d in enumerate(rows):
            labels.append(f"{d['hypothesis'].replace('H2c', 'H2-c')}: "
                          f"{fmt.FAMILY_LABEL[d['a']].replace('Diffusion U-Net', 'Diffusion')}"
                          f" − {fmt.FAMILY_LABEL[d['b']]}")
            for k, t in enumerate(TASKS):
                e = next(x for x in r["tasks"][t]["success"]["differences"]
                         if (x["a"], x["b"]) == (d["a"], d["b"]))
                y = j + (-0.15 if k == 0 else 0.15)
                est = 100 * e["difference"]
                lo, hi = 100 * e["ci95"][0], 100 * e["ci95"][1]
                filled = e["separation"] != "not separated"
                ax.errorbar([est], [y], xerr=[[est - lo], [hi - est]], fmt=MARKER[t],
                            color="#333333", mfc="#333333" if filled else "white", capsize=2,
                            ms=4, lw=1, label=t.capitalize() if j == 0 else None)
                data.setdefault(t, []).append({"a": d["a"], "b": d["b"],
                                               "difference": round(est, 4),
                                               "ci95": [round(lo, 4), round(hi, 4)],
                                               "separation": e["separation"]})
        ax.axvline(0, color="#999999", lw=0.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("paired difference in success (percentage points)")
        ax.legend(loc="center right", frameon=False)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    return fig, data


def robustness(src: Sources) -> tuple[Figure, dict[str, Any]]:
    """H4: success per family on the base starts and each perturbation (validation regime)."""
    a = src.json("l1_robustness")
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True,
                                 gridspec_kw={"width_ratios": [3, 2]})
        data: dict[str, Any] = {}
        for ax, t in zip(axes, TASKS, strict=True):
            sets = a["tasks"][t]["h4"]["sets"]
            fams = a["tasks"][t]["h4"]["bootstrap"]["families"]
            data[t] = {}
            width = 0.8 / len(LEARNED)
            for i, f in enumerate(LEARNED):
                for j, st in enumerate(sets):
                    v = fams[f"{f}:{st}"]
                    est, e = _err(v)
                    x = j - 0.4 + width * (i + 0.5)
                    ax.errorbar([x], [est], yerr=[[e[0]], [e[1]]], fmt="o", color=COLOR[f],
                                capsize=2, ms=4, lw=1,
                                label=fmt.FAMILY_LABEL[f] if (j == 0 and t == "lift") else None)
                    data[t][f"{f}:{st}"] = {"estimate": _p(v["estimate"]),
                                            "ci95": [_p(x_) for x_ in v["ci95"]]}
            ax.set_xticks(range(len(sets)))
            ax.set_xticklabels([fmt.SET_LABEL[s] for s in sets])
            ax.set_title(f"{t.capitalize()} (fresh starts, validation regime)")
            ax.set_ylim(0, 105)
            ax.grid(axis="y", lw=0.3, alpha=0.5)
        axes[0].set_ylabel("success (%)")
        axes[0].legend(loc="lower left", frameon=False, ncol=2)
    return fig, data


def latency_success(src: Sources) -> tuple[Figure, dict[str, Any]]:
    """Level 1: held-out success against median per-decision latency (canonical container)."""
    r = src.json("l1_results")
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        data: dict[str, Any] = {}
        for t in TASKS:
            data[t] = {}
            for f in LEARNED:
                lat = r["tasks"][t]["cost_latency"][f]["latency_ms_median"]
                v = r["tasks"][t]["success"]["families"][f]
                est, e = _err(v)
                ax.errorbar([lat], [est], yerr=[[e[0]], [e[1]]], fmt=MARKER[t], color=COLOR[f],
                            mfc=COLOR[f] if t == "lift" else "white", capsize=2, ms=4, lw=0.8)
                data[t][f] = {"latency_ms_median": round(lat, 4), "success": _p(v["estimate"]),
                              "ci95": [_p(x) for x in v["ci95"]]}
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("median latency per decision (ms, log scale)")
        ax.set_ylabel("held-out success (%)")
        handles = [plt.Line2D([], [], color=COLOR[f], marker="o", ls="",
                              label=fmt.FAMILY_LABEL[f]) for f in LEARNED]
        handles += [plt.Line2D([], [], color="#333333", marker=MARKER[t], ls="",
                               mfc="#333333" if t == "lift" else "white", label=t.capitalize())
                    for t in TASKS]
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6)
        ax.grid(lw=0.3, alpha=0.5)
    return fig, data


def taxonomy(src: Sources) -> tuple[Figure, dict[str, Any]]:
    """Level 1 held-out failures by preregistered category (episodes over 3 seeds)."""
    r = src.json("l1_results")
    cats = [c for c in fmt.TAXONOMY_ORDER if c != "success"]
    palette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.9), sharey=True)
        data: dict[str, Any] = {}
        for ax, t in zip(axes, TASKS, strict=True):
            data[t] = {}
            for i, f in enumerate(LEARNED):
                tax = r["tasks"][t]["taxonomy_by_family"][f]
                left = 0
                for c, col in zip(cats, palette, strict=True):
                    ax.barh([i], [tax[c]], left=left, color=col, height=0.6,
                            label=fmt.TAXONOMY_LABEL[c] if (i == 0 and t == "can") else None)
                    left += tax[c]
                data[t][f] = {c: tax[c] for c in cats}
            ax.set_yticks(range(len(LEARNED)))
            ax.set_yticklabels([fmt.FAMILY_LABEL[f] for f in LEARNED])
            ax.set_title(f"{t.capitalize()} held-out failures")
            n_ep = sum(c["n_episodes"] for c in r["tasks"][t]["per_checkpoint"]
                       if c["family"] == "diffusion_unet")
            ax.set_xlabel(f"failed episodes (of {n_ep} per family)")
        axes[0].invert_yaxis()  # shared y: invert once
        axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    return fig, data


def level2(src: Sources) -> tuple[Figure, dict[str, Any]]:
    """Level 2 placement: point cloud vs RGB diffusion on the base starts and the cube shift."""
    r = src.json("l2_results")
    fams = r["placement"]["bootstrap"]["families"]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        data: dict[str, Any] = {}
        sets = ["robustness_base", "ood_cube_shift"]
        for i, arm in enumerate(("pc_diffusion", "rgb_diffusion")):
            for j, st in enumerate(sets):
                v = fams[f"{arm}:{st}"]
                est, e = _err(v)
                x = j + (-0.12 if i == 0 else 0.12)
                ax.errorbar([x], [est], yerr=[[e[0]], [e[1]]], fmt="o", color=COLOR[arm],
                            capsize=3, ms=5, lw=1.2,
                            label=fmt.FAMILY_LABEL[arm] if j == 0 else None)
                ax.scatter([x + 0.06] * len(v["per_seed"]), [100 * s for s in v["per_seed"]],
                           s=9, facecolors="white", edgecolors=COLOR[arm], lw=0.8, zorder=4)
                data[f"{arm}:{st}"] = {"estimate": _p(v["estimate"]),
                                       "ci95": [_p(x_) for x_ in v["ci95"]],
                                       "per_seed": [_p(s) for s in v["per_seed"]]}
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([fmt.SET_LABEL[s] for s in sets])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 105)
        ax.set_ylabel("success (%)")
        ax.set_title("Lift placement (fresh starts, validation regime)")
        ax.legend(loc="lower left", frameon=False)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    return fig, data


FIGURES: dict[str, Callable[[Sources], tuple[Figure, dict[str, Any]]]] = {
    "heldout_success": heldout_success,
    "differences": differences,
    "robustness": robustness,
    "latency_success": latency_success,
    "taxonomy": taxonomy,
    "level2": level2,
}

# Genuine MuJoCo renders copied unchanged into paper/generated/figures/, located by SHA-256 in
# their directory (the file names carry state IDs, which this module does not spell out).
RENDERS: dict[str, tuple[str, str, str]] = {
    "render_lift": ("reports/session10/observatory/verification",
                    "b6a165f72de1fb92530cffab542ec441371c3aee802274e19e795596b900a33e",
                    "Lift, diffusion seed 0, held-out start 0, frame 21: MuJoCo frontview render "
                    "of "
                    "the recorded state (Session 10 viewer-verification reference)"),
    "render_can": ("reports/session10/observatory/verification",
                   "149b860b17a2eca42100036afea8081619c5b79647eb6ba86cd66b8103cd8aad",
                   "Can, diffusion seed 0, held-out start 0, frame 60: MuJoCo frontview render of "
                   "the recorded state (Session 10 viewer-verification reference)"),
    "render_pc_overlay": ("reports/session9/pc_feasibility/linux-aarch64-container-1f685638f28c",
                          "fe163485aaccd6a22cf26ced5f8de13c992544f4403563909daabcd37d72783c",
                          "Lift demo 0, t = 29: the 512 sampled cloud points projected on the "
                          "frontview and agentview frames (Session 9 feasibility, canonical "
                          "container)"),
}


def find_render(root: Any, directory: str, sha: str) -> str:
    """Repository-relative path of the PNG in ``directory`` whose SHA-256 is ``sha``."""
    hits = [p for p in sorted((root / directory).glob("*.png")) if sha256_file(p) == sha]
    if len(hits) != 1:
        raise FileNotFoundError(f"{directory}: {len(hits)} PNGs with sha256 {sha}")
    return hits[0].relative_to(root).as_posix()
