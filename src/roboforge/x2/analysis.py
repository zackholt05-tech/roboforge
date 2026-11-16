"""X2-c: the preregistered analysis of the 12 X2 held-out evaluations (preregistration §7;
DECISIONS D-130).

**Label.** "X2 (Can MH)": reported beside the frozen Level 1 Can PH result, never instead of
it, never pooled with it; within-operator claims only.

Everything here is fixed by ``docs/extensions/X2_PREREGISTRATION.md`` §7, with the
clarifications of D-130 (the relation for "neither separated" and for a conflicting verdict):

- **Outcome matrices:** per family, 3 seeds (rows 0-2) x 100 starts (the manifest's state
  order); 1 for success before the 500-step horizon, else 0. Engineering errors count as
  failures and are reported separately.
- **Primary interval:** the frozen hierarchical paired bootstrap, unchanged: B = 10,000, seed
  20261106, 95 % percentile intervals, draw order states, bc_mlp, bc_rnn, bc_mlp_chunk,
  diffusion_unet (an ENGINEERING-INCOMPLETE family is skipped).
- **Seed-level interval:** the D-095 Welch t 95 % interval on the 3 + 3 per-seed success
  rates; "undefined (no seed variance)" when both families have zero seed variance.
- **Contrasts (A - B):** H2 diffusion - BC-MLP, H2 diffusion - BC-RNN, H2c diffusion -
  BC-MLP-chunk, H3 BC-RNN - BC-MLP.
- **Verdict:** robust (+/-) when both intervals exclude 0 on the same side; bootstrap only;
  seed-level only; not separated; conflicting (no verdict).
- **Relation to Level 1 Can PH** (frozen bootstrap verdicts): the frozen bootstrap
  separation against the X2 bootstrap separation: same direction, reversed, not reproduced,
  new separation, or neither separated (D-130); "no relation" when the X2 verdict is
  conflicting.
- **Secondary (no readings):** censored episode length (first-success step, else 500) with
  the same bootstrap; per-seed rates; family means with bootstrap intervals; the frozen
  failure taxonomy; the X2-b validation selections and the latency benchmark.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from roboforge.eval.freeze import FAMILY_ORDER
from roboforge.eval.statistics import BOOTSTRAP_SEED, hierarchical_paired_bootstrap, separation
from roboforge.paper.sensitivity import welch

B = 10_000
HORIZON = 500
N_STATES = 100
SEEDS = (0, 1, 2)
CONTRASTS = (("H2", "diffusion_unet", "bc_mlp"), ("H2", "diffusion_unet", "bc_rnn"),
             ("H2c", "diffusion_unet", "bc_mlp_chunk"), ("H3", "bc_rnn", "bc_mlp"))
NAMES = {"bc_mlp": "BC-MLP", "bc_rnn": "BC-RNN", "bc_mlp_chunk": "BC-MLP-chunk",
         "diffusion_unet": "diffusion"}
HYPOTHESIS_LABEL = {"H2": "H2", "H2c": "H2-c", "H3": "H3"}
VERDICTS = ("robust", "bootstrap only", "seed-level only", "not separated", "conflicting")
RELATIONS = ("same direction", "reversed", "not reproduced", "new separation",
             "neither separated", "no relation")
UNDEFINED = "undefined (no seed variance)"
STATUS_INCOMPLETE = "not computed (ENGINEERING-INCOMPLETE family)"
FROZEN_RESULTS = ("reports/session7/results/results.json",
                  "b8b097c38765c588711615917e3cfd5be6847e63083ddebe92f0f1488672408c")
LATENCY = ("reports/session16/x2/suite/latency/linux-aarch64-container-1f685638f28c.json",
           "daea5ba211301fc7cc7c21439732c8b1858d923e668ecc0e09961a4e1ebe2e3e")


def design() -> dict[str, Any]:
    """The fixed analysis design (written into the X2 freeze; the analysis checks it)."""
    return {"bootstrap": {"function": "roboforge.eval.statistics.hierarchical_paired_bootstrap",
                          "B": B, "seed": BOOTSTRAP_SEED, "level": 0.95,
                          "interval": "percentile, numpy linear interpolation",
                          "draw_order": ["states", *FAMILY_ORDER]},
            "seed_level": {"function": "roboforge.paper.sensitivity.welch",
                           "decision": "D-095", "level": 0.95, "undefined": UNDEFINED},
            "contrasts": [{"hypothesis": h, "a": a, "b": b} for h, a, b in CONTRASTS],
            "outcome": "success before the 500-step horizon; engineering errors are failures",
            "verdicts": list(VERDICTS),
            "relation": {"against": f"{FROZEN_RESULTS[0]} (Level 1 Can PH, frozen bootstrap)",
                         "sha256": FROZEN_RESULTS[1], "labels": list(RELATIONS),
                         "rule": "frozen bootstrap separation vs X2 bootstrap separation; "
                                 "no relation when the X2 verdict is conflicting (D-130)"},
            "secondary": {"censored_episode_length": "first-success step if successful, "
                                                     "else 500; same bootstrap, 4 contrasts",
                          "descriptive": ["per-seed success rates",
                                          "family means with bootstrap intervals",
                                          "frozen failure taxonomy on every episode",
                                          "X2-b validation selections", "latency benchmark"]},
            "engineering_incomplete": "docs/FINAL_TEST_FREEZE.md §4: the family is left out of "
                                      "the bootstrap; its contrasts are not computed",
            "no_p_values": True, "multiplicity_correction": None}


def _sign(sep: str) -> str | None:
    return {"separated (+)": "+", "separated (-)": "-"}.get(sep)


def welch_separation(ci: tuple[float, float] | None) -> str:
    """The frozen separation rule on a Welch interval (undefined without seed variance)."""
    return UNDEFINED if ci is None else separation(list(ci))


def verdict(boot_sep: str, seed_sep: str) -> tuple[str, str | None]:
    """(verdict, sign) of one contrast from its two separations (preregistration §7)."""
    b, s = _sign(boot_sep), _sign(seed_sep)
    if b and s:
        return ("robust", b) if b == s else ("conflicting", None)
    if b:
        return "bootstrap only", b
    if s:
        return "seed-level only", s
    return "not separated", None


def relation(frozen_boot_sep: str, x2_boot_sep: str, x2_verdict: str) -> str:
    """Relation of the X2 bootstrap separation to the frozen Level 1 Can PH one (D-130)."""
    if x2_verdict == "conflicting":
        return "no relation"
    f, x = _sign(frozen_boot_sep), _sign(x2_boot_sep)
    if f and x:
        return "same direction" if f == x else "reversed"
    if f:
        return "not reproduced"
    return "new separation" if x else "neither separated"


def contrast_name(h: str, a: str, b: str) -> str:
    """For example ``H2 diffusion − BC-MLP``."""
    return f"{HYPOTHESIS_LABEL[h]} {NAMES[a]} − {NAMES[b]}"


def matrices(evals: list[dict[str, Any]], state_ids: list[str], families: list[str]
             ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per family: success and censored-length matrices, seeds 0-2 by rows, starts in the
    manifest's order. Each evaluation is ``{"job": ..., "summary": ...}``."""
    succ, length = {}, {}
    for fam in families:
        rows = sorted((e for e in evals if e["job"]["family"] == fam),
                      key=lambda e: e["job"]["seed"])
        if [e["job"]["seed"] for e in rows] != list(SEEDS):
            raise ValueError(f"{fam}: evaluations are not seeds {SEEDS}")
        rs, rl = [], []
        for e in rows:
            by_id = {ep["state_id"]: ep for ep in e["summary"]["episodes"]}
            if sorted(by_id) != sorted(state_ids) or len(e["summary"]["episodes"]) != len(
                    state_ids):
                raise ValueError(f"{e['job']['eval_run_id']}: episodes do not cover the "
                                 "held-out starts exactly once")
            eps = [by_id[sid] for sid in state_ids]
            rs.append([1.0 if ep["success"] else 0.0 for ep in eps])
            rl.append([float(ep["first_success_step"]) if ep["success"] else float(HORIZON)
                       for ep in eps])
        succ[fam], length[fam] = np.array(rs), np.array(rl)
    return succ, length


def _frozen_row(frozen: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    fd = next(x for x in frozen["differences"] if (x["a"], x["b"]) == (a, b))
    fci, fdf = welch(frozen["families"][a]["per_seed"], frozen["families"][b]["per_seed"])
    return {"difference": fd["difference"], "bootstrap_ci95": fd["ci95"],
            "bootstrap_separation": fd["separation"],
            "seed_level_ci95": None if fci is None else list(fci), "seed_level_df": fdf,
            "seed_level_separation": welch_separation(fci),
            "n_seeds": frozen["families"][a]["n_seeds"],
            "n_states": frozen["families"][a]["n_states"], "horizon": 400}


def analyse(succ: dict[str, np.ndarray], length: dict[str, np.ndarray],
            frozen: dict[str, Any]) -> dict[str, Any]:
    """Primary and secondary statistics on the complete families (``succ`` keys)."""
    order = [f for f in FAMILY_ORDER if f in succ]
    comps = [(a, b) for _, a, b in CONTRASTS if a in order and b in order]
    boot = hierarchical_paired_bootstrap({f: succ[f] for f in order}, comps, b=B,
                                         seed=BOOTSTRAP_SEED, order=order)
    if boot["draw_order"] != ["states", *order]:
        raise ValueError(f"draw order {boot['draw_order']}")
    diffs = {(d["a"], d["b"]): d for d in boot["differences"]}
    rows = []
    for h, a, b in CONTRASTS:
        name = contrast_name(h, a, b)
        frozen_side = _frozen_row(frozen, a, b)
        if (a, b) not in diffs:
            rows.append({"hypothesis": h, "a": a, "b": b, "name": name,
                         "status": STATUS_INCOMPLETE, "frozen": frozen_side})
            continue
        d = diffs[(a, b)]
        ci, df = welch(boot["families"][a]["per_seed"], boot["families"][b]["per_seed"])
        seed_sep = welch_separation(ci)
        v, sign = verdict(d["separation"], seed_sep)
        rows.append({"hypothesis": h, "a": a, "b": b, "name": name, "status": "computed",
                     "difference": d["difference"], "bootstrap_ci95": d["ci95"],
                     "bootstrap_separation": d["separation"],
                     "seed_level_ci95": None if ci is None else list(ci),
                     "seed_level_df": df, "seed_level_separation": seed_sep,
                     "verdict": v, "sign": sign,
                     "relation": relation(frozen_side["bootstrap_separation"],
                                          d["separation"], v),
                     "frozen": frozen_side})
    cel = hierarchical_paired_bootstrap({f: length[f] for f in order}, comps, b=B,
                                        seed=BOOTSTRAP_SEED, order=order)
    return {"families": order, "bootstrap": boot, "contrasts": rows,
            "censored_episode_length": cel}


def statement(r: dict[str, Any]) -> str:
    """One plain sentence per contrast, in the preregistered terms only."""
    if r["status"] != "computed":
        return f"{r['name']}: {r['status']}."
    v = r["verdict"] + (f" ({r['sign']})" if r["sign"] else "")
    return (f"{r['name']}: {100 * r['difference']:+.1f} pp; {v}. Relation to the frozen "
            f"Level 1 Can PH verdict ({r['frozen']['bootstrap_separation']}): "
            f"{r['relation']}. X2 (Can MH), within-operator; never pooled with Level 1.")


def _pp(x: float) -> str:
    return f"{100 * x:+.1f}"


def _ci(ci: list[float] | None) -> str:
    return "undefined" if ci is None else f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def tables(res: dict[str, Any]) -> str:
    """The generated ``tables.md`` (every number from ``results.json``)."""
    p = res["primary"]
    lines = ["# X2 (Can MH) results: the X2 held-out suite, once", "",
             "Generated by `scripts/analyze_x2c.py`; every number comes from `results.json`. "
             "X2 (Can MH) is reported beside the frozen Level 1 Can PH result, never instead "
             "of it and never pooled with it. Within-operator claims only (D-119). X2 "
             "episodes end at 500 steps, Level 1 Can PH at 400, so the two success scales "
             "differ (D-120).", "",
             "## Declared contrasts: Level 1 Can PH (frozen) beside X2 (Can MH)", "",
             "| Contrast | Level 1 Can PH (3 seeds × 50, 400 steps): difference, bootstrap "
             "95 %, verdict | Level 1 seed-level 95 % | X2: difference | X2 bootstrap 95 % "
             "| X2 seed-level 95 % (df) | X2 verdict | Relation |",
             "|---|---|---|---|---|---|---|---|"]
    for r in p["contrasts"]:
        f = r["frozen"]
        fz = (f"{_pp(f['difference'])} pp {_ci(f['bootstrap_ci95'])}, "
              f"{f['bootstrap_separation']} | {_ci(f['seed_level_ci95'])}")
        if r["status"] != "computed":
            lines.append(f"| {r['name']} | {fz} | | | | {r['status']} | |")
            continue
        df = "" if r["seed_level_df"] is None else f" ({r['seed_level_df']:.1f})"
        v = r["verdict"] + (f" ({r['sign']})" if r["sign"] else "")
        lines.append(f"| {r['name']} | {fz} | {_pp(r['difference'])} pp | "
                     f"{_ci(r['bootstrap_ci95'])} | {_ci(r['seed_level_ci95'])}{df} | {v} | "
                     f"{r['relation']} |")
    lines += ["", "## Statements", ""] + [f"- {statement(r)}" for r in p["contrasts"]]
    eng = res["engineering"]
    lines += ["", "## Engineering status", "",
              "Families: " + ", ".join(f"{NAMES[f]} {s}" for f, s in eng["families"].items())
              + f". Incomplete evaluations: {len(eng['incomplete'])}. Episodes with an "
              f"engineering error (counted as failures): {eng['n_errors_total']}. Second "
              f"attempts: {eng['n_attempt2']}."]
    lines += ["", "## Family success on the X2 held-out starts (3 seeds × 100)", "",
              "| Family | Success % | Bootstrap 95 % | Per-seed % (seeds 0–2) | Level 1 Can PH "
              "(3 × 50) % |", "|---|---|---|---|---|"]
    for fam in p["families"]:
        fb = p["bootstrap"]["families"][fam]
        per = ", ".join(f"{100 * v:.0f}" for v in fb["per_seed"])
        lines.append(f"| {NAMES[fam]} | {100 * fb['estimate']:.1f} | "
                     f"[{100 * fb['ci95'][0]:.1f}, {100 * fb['ci95'][1]:.1f}] | {per} | "
                     f"{100 * res['frozen_families'][fam]['estimate']:.1f} |")
    lines += ["", "## Secondary (declared; no readings)", "",
              "**Censored episode length** (steps; first success, else 500).", "",
              "| Contrast | Difference | Bootstrap 95 % |", "|---|---|---|"]
    hyp = {(a, b): h for h, a, b in CONTRASTS}
    for d in p["censored_episode_length"]["differences"]:
        lines.append(f"| {contrast_name(hyp[(d['a'], d['b'])], d['a'], d['b'])} | "
                     f"{d['difference']:+.1f} | [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}] "
                     f"{d['separation']} |")
    lines += ["", "**Evaluations.**", "",
              "| Family | Seed | Training run | Step | Successes / 100 | Wilson 95 % | Errors "
              "| X2-b validation / 50 | Latency median (ms) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for e in res["evaluations"]:
        lat = "" if e["latency_ms_median"] is None else f"{e['latency_ms_median']:.2f}"
        lines.append(f"| {NAMES[e['family']]} | {e['seed']} | `{e['train_run_id']}` | "
                     f"{e['step']} | {e['successes']} | [{100 * e['wilson95'][0]:.1f}, "
                     f"{100 * e['wilson95'][1]:.1f}] | {e['n_errors']} | "
                     f"{e['validation_successes']} | {lat} |")
    cats = res["taxonomy_categories"]
    lines += ["", "**Failure taxonomy** (episodes, 3 seeds × 100).", "",
              "| Family | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for fam in FAMILY_ORDER:
        c = res["taxonomy"].get(fam)
        if c is not None:
            lines.append(f"| {NAMES[fam]} | " + " | ".join(str(c[k]) for k in cats) + " |")
    return "\n".join(lines) + "\n"
