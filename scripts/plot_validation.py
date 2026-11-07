"""Validation-only figures and table of a main suite: curves, seed spread, actions, latency.

    uv run --frozen python scripts/plot_validation.py \
        --summary reports/session6/suite/level1_core_lift.json \
        --latency reports/session6/cost_latency/linux-aarch64-container-1f685638f28c.json

Reads only machine-readable artifacts: the suite summary (``scripts/run_suite.py``), every
run's manifest and ``metrics.jsonl``, the canonical validation summaries, the selection and
tuning records, the selected evaluations' episode arrays, the committed training
demonstrations (actions only, for reference) and an optional latency report
(``scripts/benchmark_policies.py``, canonical container, run alone). It writes to
``--out-dir`` (default ``<reports>/figures``):

* ``training_curves.png``: training loss and the offline validation metric per run. Offline
  metrics never select (RESEARCH_PROTOCOL §7.4);
* ``validation_success.png``: successes out of 50 at every checkpoint of every run, with the
  §7-selected checkpoint marked;
* ``seed_spread.png``: the selected checkpoints of the chosen configuration, seeds 0-2;
* ``action_histograms.png``: executed actions of those selected checkpoints against the
  training demonstrations; motion dimensions and the gripper are reported separately
  (KI-022);
* ``success_vs_latency.png``: mean success over seeds against the median decision latency;
* ``validation_table.md``: the same numbers as a table, generated, never typed;
* ``index.json``: every figure's and source's SHA-256, the git revision and the time.

Every output is labelled VALIDATION ONLY: checkpoints and configurations were selected on the
same 50 validation starts, so these are development numbers, never the held-out test.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from roboforge.data.sequence import SequenceDataset  # noqa: E402
from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import git_revision, sha256_file  # noqa: E402

LABEL = ("VALIDATION ONLY: the 50 validation starts of the task; checkpoints and "
         "configurations were selected on the same starts. Development numbers, not the "
         "held-out test.")
# Categorical slots 1-4 of the validated reference palette (dataviz), one per trial; a
# replicate seed reuses its trial's colour with a dashed line (secondary encoding).
TRIAL_COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
FAMILY_LABEL = {"diffusion_unet": "Diffusion U-Net", "bc_mlp": "BC-MLP",
                "bc_mlp_chunk": "BC-MLP-chunk", "bc_rnn": "BC-RNN"}
FAMILY_MARKER = {"diffusion_unet": "o", "bc_mlp": "s", "bc_mlp_chunk": "D", "bc_rnn": "^"}
DIM_NAMES = ["Δx", "Δy", "Δz", "Δrx", "Δry", "Δrz", "gripper"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


class Sources:
    """Every input file read, with its SHA-256 (the figures' provenance)."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def json(self, path: Path) -> Any:
        self.files[repo_relative(path)] = sha256_file(path)
        return json.loads(path.read_text())

    def lines(self, path: Path) -> list[dict[str, Any]]:
        self.files[repo_relative(path)] = sha256_file(path)
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _style(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _footer(fig: plt.Figure, src: str, git: dict[str, Any], when: str) -> None:
    """Validation-only label, source and revision, wrapped to the figure width (about 17
    characters per inch at 6.5 pt) so that no figure can clip the label."""
    rev = f"{git['sha'][:8]}{'-dirty' if git['dirty'] else ''}"
    label = textwrap.fill(LABEL, width=max(60, int(fig.get_figwidth() * 17)))
    fig.text(0.01, 0.005, f"{label}\nsource: {src}\ngit {rev} · generated {when}",
             fontsize=6.5, color=MUTED, va="bottom")


def _dash(r: dict[str, Any]) -> str:
    """Solid for a tuning trial; dashed (seed 1) or dotted (seed 2) for a replicate."""
    return "-" if r["role"] == "tuning" else ("--" if r["seed"] == 1 else ":")


def _run_label(r: dict[str, Any]) -> str:
    if r["role"] == "tuning":
        return f"trial {r['trial']}"
    return f"trial {r['trial']}, seed {r['seed']}"


def load(args: argparse.Namespace, src: Sources) -> dict[str, Any]:
    """Suite summary plus each run's manifest, metrics and per-checkpoint validation."""
    summary = src.json(args.summary)
    for fam in summary["families"]:
        if fam["tuning_record"]:
            fam["tuning"] = src.json(REPO_ROOT / fam["tuning_record"])
        for r in fam["runs"]:
            run_dir = REPO_ROOT / "runs" / r["run_id"]
            r["manifest"] = src.json(run_dir / "manifest.json")
            r["metrics"] = src.lines(run_dir / "metrics.jsonl")
            r["evals"] = {}
            if r["manifest"]["status"] != "complete":
                continue
            for c in r["manifest"]["checkpoints"]:
                hits = sorted((args.reports_dir / "validation").glob(
                    f"{r['run_id']}__val_step{c['step']:07d}__*.json"))
                if len(hits) != 1:
                    raise SystemExit(f"{r['run_id']} step {c['step']}: {len(hits)} summaries")
                r["evals"][c["step"]] = src.json(hits[0])
    return summary


def chosen_runs(fam: dict[str, Any]) -> list[dict[str, Any]]:
    """Seed 0 of the chosen trial plus its replicates, by seed."""
    t = fam["chosen_trial"]
    return sorted([r for r in fam["runs"] if r["trial"] == t and r["selection"]],
                  key=lambda r: r["seed"])


def fig_training(summary: dict[str, Any], out: Path, foot: tuple) -> Path:
    fams = summary["families"]
    fig, axes = plt.subplots(len(fams), 2, figsize=(11, 2.9 * len(fams)), squeeze=False)
    for i, fam in enumerate(fams):
        for r in fam["runs"]:
            colour = TRIAL_COLOURS[r["trial"]]
            ls = _dash(r)
            tr = [m for m in r["metrics"] if m["kind"] == "train"]
            off = [m for m in r["metrics"] if m["kind"] == "offline"]
            axes[i, 0].plot([m["step"] for m in tr], [m["loss"] for m in tr], color=colour,
                            ls=ls, lw=1.2, label=_run_label(r))
            key = "denoising_mse" if fam["family"] == "diffusion_unet" else "mse"
            axes[i, 1].plot([m["step"] for m in off], [m["valid"][key] for m in off],
                            color=colour, ls=ls, lw=1.2, label=_run_label(r))
        axes[i, 0].set_yscale("log")
        name = FAMILY_LABEL.get(fam["family"], fam["family"])
        noise = ", noise" if fam["family"] == "diffusion_unet" else ""
        axes[i, 0].set_title(f"{name}: training loss (masked MSE{noise})", fontsize=9,
                             color=INK, loc="left")
        what = ("noise MSE, EMA" if fam["family"] == "diffusion_unet" else "action MSE")
        axes[i, 1].set_title(f"{name}: offline valid-demo {what} (never selects)",
                             fontsize=9, color=INK, loc="left")
        for ax in axes[i]:
            _style(ax)
            ax.set_xlabel("training step", fontsize=8, color=MUTED)
        axes[i, 1].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _footer(fig, *foot)
    p = out / "training_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_success(summary: dict[str, Any], out: Path, foot: tuple) -> Path:
    fams = summary["families"]
    fig, axes = plt.subplots(len(fams), 1, figsize=(9, 2.6 * len(fams)), squeeze=False)
    for i, fam in enumerate(fams):
        ax = axes[i, 0]
        for r in fam["runs"]:
            if not r["evals"]:
                continue
            steps = sorted(r["evals"])
            ax.plot(steps, [r["evals"][s]["successes"] for s in steps],
                    color=TRIAL_COLOURS[r["trial"]], ls=_dash(r),
                    lw=1.4, marker="o", ms=3, label=_run_label(r))
            sel = r["selection"]
            ax.plot([sel["step"]], [sel["successes"]], marker="o", ms=10, mfc="none",
                    mec=TRIAL_COLOURS[r["trial"]], mew=1.6)
        name = FAMILY_LABEL.get(fam["family"], fam["family"])
        ax.set_title(f"{name}: validation successes out of 50 per checkpoint (ring = §7 "
                     f"selection; chosen configuration: trial {fam['chosen_trial']})",
                     fontsize=9, color=INK, loc="left")
        ax.set_ylim(min(30, ax.get_ylim()[0]), 50.9)
        ax.set_ylabel("successes / 50", fontsize=8, color=MUTED)
        ax.set_xlabel("checkpoint step", fontsize=8, color=MUTED)
        _style(ax)
        ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower right")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    _footer(fig, *foot)
    p = out / "validation_success.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_seed_spread(summary: dict[str, Any], out: Path, foot: tuple) -> Path:
    fams = summary["families"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle("Selected checkpoints of each family's chosen configuration, seeds 0-2",
                 fontsize=9, color=INK, x=0.01, ha="left")
    for j, fam in enumerate(fams):
        runs = chosen_runs(fam)
        succ = [r["selection"]["successes"] for r in runs]
        fs = [r["selection"]["first_success_step_mean"] for r in runs]
        for ax, vals in ((axes[0], succ), (axes[1], fs)):
            xs = j + np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(xs, vals, s=40, marker=FAMILY_MARKER.get(fam["family"], "o"),
                       color=TRIAL_COLOURS[fam["chosen_trial"]], edgecolor="white", lw=1,
                       zorder=3)
            good = [v for v in vals if v is not None]
            if good:
                ax.hlines(np.mean(good), j - 0.25, j + 0.25, color=INK, lw=1.4, zorder=2)
            for x, v, r in zip(xs, vals, runs, strict=True):
                if v is not None:
                    ax.annotate(f"s{r['seed']}", (x, v), xytext=(4, 2),
                                textcoords="offset points", fontsize=6.5, color=MUTED)
    names = [f"{FAMILY_LABEL.get(f['family'], f['family'])}\n(trial {f['chosen_trial']})"
             for f in fams]
    for ax, title, yl in ((axes[0], "Successes / 50", "successes"),
                          (axes[1], "Mean first-success step", "control steps (20 Hz)")):
        ax.set_xticks(range(len(fams)), names, fontsize=8)
        ax.set_title(title + " (bar: seed mean)", fontsize=9, color=INK, loc="left")
        ax.set_ylabel(yl, fontsize=8, color=MUTED)
        _style(ax)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    _footer(fig, *foot)
    p = out / "seed_spread.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def episode_actions(eval_run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Executed actions and clip masks of every episode of one evaluation run."""
    acts, clips = [], []
    for npz in sorted((REPO_ROOT / "runs" / eval_run_id / "episodes").glob("*/arrays.npz")):
        a = np.load(npz)
        acts.append(a["executed_action"])
        clips.append(a["clip_mask"])
    return np.concatenate(acts), np.concatenate(clips)


def fig_actions(summary: dict[str, Any], task: str, out: Path, foot: tuple,
                src: Sources) -> tuple[Path, dict[str, Any]]:
    ds = SequenceDataset.from_committed(task, "train", obs_horizon=1, pred_horizon=1)
    demo = ds.action_normalizer.unnormalize(np.concatenate([ds._act[d] for d in ds.demos]))
    fams = summary["families"]
    fig, axes = plt.subplots(len(fams), 7, figsize=(15, 2.2 * len(fams)), squeeze=False)
    clip_stats: dict[str, Any] = {}
    bins = np.linspace(-1, 1, 41)
    for i, fam in enumerate(fams):
        runs = chosen_runs(fam)
        acts, clips = zip(*(episode_actions(r["selection"]["eval_run_id"]) for r in runs),
                          strict=True)
        a, c = np.concatenate(acts), np.concatenate(clips)
        clip_stats[fam["family"]] = {
            "n_steps": int(len(a)), "motion_clip_fraction": float(c[:, :6].mean()),
            "gripper_clip_fraction": float(c[:, 6].mean()),
            "runs": [r["selection"]["eval_run_id"] for r in runs]}
        for d in range(7):
            ax = axes[i, d]
            ax.hist(demo[:, d], bins=bins, density=True, color="#c3c2b7",
                    label="demonstrations (train split)")
            ax.hist(a[:, d], bins=bins, density=True, histtype="step", lw=1.4,
                    color=TRIAL_COLOURS[fam["chosen_trial"]], label="policy, executed")
            ax.set_title(DIM_NAMES[d], fontsize=8, color=INK)
            _style(ax)
            ax.set_yticks([])
        name = FAMILY_LABEL.get(fam["family"], fam["family"])
        cs = clip_stats[fam["family"]]
        axes[i, 0].set_ylabel(f"{name}\nmotion clip {100 * cs['motion_clip_fraction']:.1f} %\n"
                              f"gripper clip {100 * cs['gripper_clip_fraction']:.1f} %",
                              fontsize=7.5, color=INK)
    axes[0, 6].legend(fontsize=6.5, frameon=False, loc="upper center")
    fig.suptitle("Executed actions of the selected checkpoints (chosen configuration, seeds "
                 "0-2, 50 validation episodes each) vs demonstrations; normalized env units "
                 "[−1, 1]. Gripper clipping is harmless (KI-022)", fontsize=9, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    _footer(fig, *foot)
    p = out / "action_histograms.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, clip_stats


def latency_by_family(summary: dict[str, Any], latency: dict[str, Any] | None
                      ) -> dict[str, dict[str, Any]]:
    """Median observe+act latency per family from the benchmark report."""
    if latency is None:
        return {}
    out = {}
    for fam in summary["families"]:
        ids = {r["run_id"] for r in fam["runs"]}
        rows = [row for row in latency["rows"] if row["train_run_id"] in ids]
        if rows:
            inf = rows[0]["inference"]
            out[fam["family"]] = {"train_run_id": rows[0]["train_run_id"],
                                  "median_ms": inf["observe_plus_act_ms"]["median"],
                                  "p95_ms": inf["observe_plus_act_ms"]["p95"],
                                  "platform_id": latency["platform"]["platform_id"]}
    return out


def fig_latency(summary: dict[str, Any], lat: dict[str, dict[str, Any]], out: Path,
                foot: tuple) -> Path | None:
    if not lat:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = []
    for fam in summary["families"]:
        if fam["family"] not in lat:
            continue
        succ = [r["selection"]["successes"] for r in chosen_runs(fam)]
        x = lat[fam["family"]]["median_ms"]
        xs.append(x)
        m = float(np.mean(succ))
        ax.errorbar([x], [m], yerr=[[m - min(succ)], [max(succ) - m]], fmt="none",
                    ecolor=MUTED, lw=1, capsize=3)
        ax.scatter([x], [m], s=60, marker=FAMILY_MARKER[fam["family"]], color=INK, zorder=3)
        ax.annotate(f"{FAMILY_LABEL[fam['family']]} (trial {fam['chosen_trial']})", (x, m),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(min(xs) / 1.6, max(xs) * 5)  # room for the right-most label
    ax.set_xlabel("median decision latency, ms (observe + act, 1 thread)\ncanonical "
                  "container, run alone; final checkpoint of the seed-0 run", fontsize=8,
                  color=MUTED)
    ax.set_ylabel("mean successes / 50 over seeds\n(bars: min-max seed)", fontsize=8,
                  color=MUTED)
    ax.set_title("Validation success of the selected checkpoints vs inference latency",
                 fontsize=9, color=INK, loc="left")
    _style(ax)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    _footer(fig, *foot)
    p = out / "success_vs_latency.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def table(summary: dict[str, Any], clip: dict[str, Any], lat: dict[str, Any],
          src_note: str) -> str:
    """Markdown table of the tuning trials and the chosen configuration's seeds."""
    lines = [f"<!-- generated by scripts/plot_validation.py from {src_note}; do not edit -->",
             f"**{LABEL}**", "",
             "| Family | Trial | Seed | Role | Selected step | Successes / 50 | Wilson 95 % | "
             "Mean first-success step | Selected checkpoint SHA-256 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for fam in summary["families"]:
        for r in fam["runs"]:
            sel = r["selection"]
            if sel is None:
                lines.append(f"| {fam['family']} | {r['trial']} | {r['seed']} | {r['role']} | "
                             f"— | run {r['status']} | — | — | — |")
                continue
            w = r["evals"][sel["step"]]["wilson95"]
            mark = " (chosen)" if r["trial"] == fam["chosen_trial"] else ""
            lines.append(f"| {fam['family']} | {r['trial']}{mark} | {r['seed']} | {r['role']} | "
                         f"{sel['step']} | {sel['successes']} | {w[0]:.2f}–{w[1]:.2f} | "
                         f"{sel['first_success_step_mean']:.1f} | `{sel['sha256'][:12]}…` |"
                         if sel["first_success_step_mean"] is not None else
                         f"| {fam['family']} | {r['trial']}{mark} | {r['seed']} | {r['role']} | "
                         f"{sel['step']} | {sel['successes']} | {w[0]:.2f}–{w[1]:.2f} | — | "
                         f"`{sel['sha256'][:12]}…` |")
    lines += ["", "| Family | Chosen trial | Mean successes over seeds | Per seed | Motion clip "
              "(dims 0–5) | Gripper clip (dim 6) | Median latency (ms) |",
              "|---|---|---|---|---|---|---|"]
    for fam in summary["families"]:
        runs = chosen_runs(fam)
        s = [r["selection"]["successes"] for r in runs]
        c = clip.get(fam["family"], {})
        la = lat.get(fam["family"])
        la_txt = "—" if la is None else f"{la['median_ms']:.2f}"
        lines.append(f"| {fam['family']} | {fam['chosen_trial']} | {np.mean(s):.2f} | "
                     f"{', '.join(map(str, s))} | "
                     f"{100 * c.get('motion_clip_fraction', float('nan')):.2f} % | "
                     f"{100 * c.get('gripper_clip_fraction', float('nan')):.1f} % | "
                     f"{la_txt} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--reports-dir", type=Path, default=None,
                   help="default: the directory above the summary's suite/ directory")
    p.add_argument("--latency", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    a = p.parse_args(argv)
    a.reports_dir = a.reports_dir or a.summary.parent.parent
    out = a.out_dir or a.reports_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    src = Sources()
    summary = load(a, src)
    lat = latency_by_family(summary, src.json(a.latency) if a.latency else None)
    git, when = git_revision(), datetime.now(UTC).isoformat(timespec="seconds")
    foot = (repo_relative(a.summary), git, when)
    figs = [fig_training(summary, out, foot), fig_success(summary, out, foot),
            fig_seed_spread(summary, out, foot)]
    pa, clip = fig_actions(summary, summary["task"], out, foot, src)
    figs.append(pa)
    pl = fig_latency(summary, lat, out, foot)
    if pl:
        figs.append(pl)
    tab = out / "validation_table.md"
    tab.write_text(table(summary, clip, lat, repo_relative(a.summary)))
    figs.append(tab)
    index = {"schema": "roboforge.validation_figures/1", "label": LABEL,
             "outputs": {repo_relative(f): sha256_file(f) for f in figs},
             "clip_rates": clip, "latency": lat, "sources": src.files,
             "demonstrations": "committed train split actions (SequenceDataset.from_committed)",
             "git": git, "generated_utc": when}
    (out / "index.json").write_text(json.dumps(index, indent=1, allow_nan=False) + "\n")
    for f in figs:
        print(repo_relative(f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
