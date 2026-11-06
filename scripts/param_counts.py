"""Exact parameter counts of the primary configurations and capacity-matched tiers.

    python scripts/param_counts.py

Counts are taken from the instantiated networks (``count_parameters``), never computed by
hand. For each target budget (0.27 M, 1.08 M = the primary tier, 4.3 M) the script finds
the width that brings BC-MLP (2 hidden layers), BC-RNN (2-layer LSTM and GRU) and
BC-MLP-chunk closest to the target. A later session can then match diffusion's parameter
count (RESEARCH_PROTOCOL §6.1). Writes ``<out>`` (default
``reports/session$ROBOFORGE_EVIDENCE_SESSION/param_counts.json``, D-039).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.models.registry import build_model, count_parameters
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative
from roboforge.provenance import git_revision
from roboforge.train.config import load_config

OBS_DIM, ACTION_DIM = 19, 7  # both tasks (D-018)
TARGETS = (269_314, 1_077_255, 4_309_020)  # 1/4x, 1x, 4x the primary BC-MLP


def count(family: str, model: dict, t_o: int, t_p: int) -> int:
    """Parameters of one instantiated network."""
    return count_parameters(build_model(family, model, OBS_DIM, ACTION_DIM, t_o, t_p))


def matched(family: str, base: dict, key: str, t_o: int, t_p: int, target: int) -> dict:
    """Width closest to ``target`` parameters (monotone in width; bisection)."""
    def with_width(w: int) -> dict:
        m = dict(base)
        m[key] = [w] * len(base[key]) if isinstance(base[key], list) else w
        return m
    lo, hi = 4, 8192
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if count(family, with_width(mid), t_o, t_p) <= target else (lo, mid)
    best = min((lo, hi), key=lambda w: abs(count(family, with_width(w), t_o, t_p) - target))
    n = count(family, with_width(best), t_o, t_p)
    return {"width": best, "params": n, "rel_diff": n / target - 1}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    args.out = args.out or evidence_reports_dir("param_counts.json")
    primary = {}
    for name in ("bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet"):
        cfg = load_config(REPO_ROOT / "configs" / "models" / f"{name}.yaml")
        primary[name] = {"family": cfg.family, "model": cfg.model,
                         "horizons": vars(cfg.horizons),
                         "params": count(cfg.family, cfg.model, cfg.horizons.obs,
                                         cfg.horizons.pred)}
    base = primary["bc_mlp"]["params"]
    for v in primary.values():
        v["rel_diff_vs_bc_mlp"] = v["params"] / base - 1
    mlp, rnn = primary["bc_mlp"]["model"], primary["bc_rnn"]["model"]
    tiers = {str(t): {
        "bc_mlp (T_o=1,T_p=1)": matched("bc_mlp", mlp, "hidden", 1, 1, t),
        "bc_rnn lstm (L=10,T_p=1)": matched("bc_rnn", rnn, "hidden", 10, 1, t),
        "bc_rnn gru (L=10,T_p=1)": matched("bc_rnn", {**rnn, "cell": "gru"}, "hidden", 10, 1, t),
        "bc_mlp_chunk (T_o=2,T_p=16)": matched("bc_mlp", mlp, "hidden", 2, 16, t)}
        for t in TARGETS}
    d = primary["diffusion_unet"]
    net = build_model(d["family"], d["model"], OBS_DIM, ACTION_DIM, d["horizons"]["obs"],
                      d["horizons"]["pred"])
    d["breakdown"] = {k: count_parameters(getattr(net.unet, k)) for k in (
        "diffusion_step_encoder", "down_modules", "mid_modules", "up_modules", "final_conv")}
    d["breakdown"]["observation_encoder"] = count_parameters(net.encoder)
    budget = primary["bc_mlp"]["params"]
    capacity_rule = {"rule": "RESEARCH_PROTOCOL §6.1: if diffusion exceeds 2x the 1.08 M BC "
                             "budget, add capacity-matched BC-MLP and BC-RNN",
                     "bc_budget_params": budget, "threshold_params": 2 * budget,
                     "diffusion_params": d["params"], "ratio": d["params"] / budget,
                     "triggered": d["params"] > 2 * budget}
    report = {"schema": "roboforge.param_counts/1", "obs_dim": OBS_DIM,
              "action_dim": ACTION_DIM, "primary": primary, "matched_tiers": tiers,
              "capacity_rule": capacity_rule,
              "git": git_revision()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for k, v in primary.items():
        print(f"{k:14s} {v['params']:>10,d}  ({v['rel_diff_vs_bc_mlp']:+.1%} vs bc_mlp)")
    for t, rows in tiers.items():
        print(f"target {int(t):,d}: " + "; ".join(f"{k} w={r['width']} {r['params']:,d} "
                                                   f"({r['rel_diff']:+.2%})"
                                                   for k, r in rows.items()))
    print(f"-> {repo_relative(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
