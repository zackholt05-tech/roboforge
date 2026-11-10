"""Declared secondary metrics that no frozen analysis reported (audit C-9; DECISIONS D-096).

RESEARCH_PROTOCOL section 8 declares gripper sign changes and action smoothness as secondary,
descriptive metrics. Every Level 1 held-out evaluation summary computes them per episode, but
the frozen results never aggregated them. This module reads the 26 held-out summaries that
``reports/session7/results/results.json`` cites, each verified against the SHA-256 recorded
there, and reports plain means. It computes no interval or test and is labelled "declared
secondary, aggregated after the frozen analysis".

Definitions (``roboforge.eval.validation``): gripper sign changes counts changes of the sign of
the executed gripper command within an episode; action smoothness is the mean absolute
step-to-step change of the executed command, here averaged over the six motion dimensions.
Episodes end at first success, so both depend on episode length.
"""

from __future__ import annotations

from typing import Any

from roboforge.paper import fmt
from roboforge.paper.sources import Sources

TASKS = ["lift", "can"]
ORDER = ["bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet", "random_uniform"]
MOTION = slice(0, 6)


def _mean(x: list[float]) -> float:
    return sum(x) / len(x)


def rows(src: Sources) -> dict[str, dict[str, dict[str, Any]]]:
    """``{task: {family: metrics}}`` aggregated over every held-out episode of the family."""
    r = src.json("l1_results")
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for t in TASKS:
        fams: dict[str, dict[str, Any]] = {}
        for pc in r["tasks"][t]["per_checkpoint"]:
            d = src.chained_json("l1_results", pc["summary"], pc["summary_sha256"])
            eps = d["episodes"]
            sign = [float(e["gripper_sign_changes"]) for e in eps]
            delta = [_mean(e["mean_abs_action_delta"][MOTION]) for e in eps
                     if e.get("mean_abs_action_delta") is not None]
            f = fams.setdefault(pc["family"], {"seeds": [], "sign": [], "delta": [],
                                               "clip": []})
            f["seeds"].append({"sign": _mean(sign), "delta": _mean(delta)})
            f["sign"] += sign
            f["delta"] += delta
            f["clip"].append(pc["motion_clip_fraction"])
        for f in fams.values():
            f["n"] = len(f["sign"])
            f["sign_mean"] = _mean(f["sign"])
            f["delta_mean"] = _mean(f["delta"])
            f["clip_mean"] = _mean(f["clip"])
            f["sign_range"] = (min(s["sign"] for s in f["seeds"]),
                               max(s["sign"] for s in f["seeds"]))
        out[t] = fams
    return out


def table(src: Sources) -> str:
    """LaTeX tabular of the declared secondary metrics on the held-out starts."""
    data = rows(src)
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Task & Family & Episodes & Gripper sign changes & Mean $|\\Delta a|$ (motion) & "
             "Clipped motion actions (\\%) \\\\",
             "\\midrule"]
    for i, t in enumerate(TASKS):
        if i:
            lines.append("\\addlinespace")
        for fam in ORDER:
            f = data[t].get(fam)
            if f is None:
                continue
            lo, hi = f["sign_range"]
            rng = "" if len(f["seeds"]) == 1 else f" [{lo:.2f}, {hi:.2f}]"
            lines.append(f"{t.capitalize()} & {fmt.FAMILY_LABEL[fam]} & {f['n']} & "
                         f"{f['sign_mean']:.2f}{rng} & {f['delta_mean']:.3f} & "
                         f"{fmt.pct(f['clip_mean'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def numbers(src: Sources, n: Any) -> None:
    """``sec.<task>.<family>.*`` macros and the number of summaries read."""
    data = rows(src)
    total = 0
    for t in TASKS:
        for fam, f in data[t].items():
            k = f"sec.{t}.{fmt.FAMILY_KEY[fam]}"
            n.set(f"{k}.sign", f"{f['sign_mean']:.2f}")
            n.set(f"{k}.delta", f"{f['delta_mean']:.3f}")
            n.set(f"{k}.clip", fmt.pct(f["clip_mean"]))
            total += len(f["seeds"])
    n.set("sec.nsummaries", total)
