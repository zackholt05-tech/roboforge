"""Every value of the v1.1 addendum as an ``\\ax{key}`` macro, from the pinned results (D-133).

Values are copied at their tables' rounding with the paper's formatting helpers
(:mod:`roboforge.paper.fmt`), so each one matches the study's own ``tables.md`` digit for
digit. A few values are derived by plain arithmetic or counting; they are listed in
:data:`DERIVED` and in the generated index. Nothing is estimated or recomputed.

Key scheme (letters, digits, ``.`` and ``-``):

- ``x1.*``: X1, exploratory, after the freeze (validation regime, Session 7 robustness sets);
- ``x3.*``: X3, separate replication (Can, fresh held-out starts);
- ``x2.*``: X2 (Can MH), beside the frozen Level 1 Can PH result, never pooled;
- ``l1.*``: the frozen Level 1 Can PH values that the X3 and X2 files carry for comparison.
"""

from __future__ import annotations

import re
from typing import Any

from roboforge.paper import fmt
from roboforge.paper.numbers import Numbers

FAMILIES = ("bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet")
# The declared Can contrasts of X3 and X2, by the name both files use.
CONTRASTS = {
    "h2bcmlp": "H2 diffusion − BC-MLP",
    "h2bcrnn": "H2 diffusion − BC-RNN",
    "h2c": "H2-c diffusion − BC-MLP-chunk",
    "h3": "H3 BC-RNN − BC-MLP",
}
# X1 arms on the robustness sets: the frozen family or its clamped re-evaluation.
X1_ARMS = {
    "diff": "diffusion_unet", "chunk": "bc_mlp_chunk", "chunkc": "bc_mlp_chunk+clamp",
    "bcmlp": "bc_mlp", "bcmlpc": "bc_mlp+clamp", "bcrnn": "bc_rnn", "bcrnnc": "bc_rnn+clamp",
}
SETS = {"robustness_base": "base", "ood_cube_shift": "cube", "ood_arm_offset": "arm"}
DERIVED = {
    "x1.nevals": "number of X1 evaluations listed in results.json (both tasks)",
    "x1.cu.n": "count of clamped - unclamped differences (same checkpoints), both tasks",
    "x1.cu.nsep": "how many of them are separated",
    "x1.cu.maxabs": "largest absolute clamped - unclamped difference, in pp",
    "x1.kept.n": "drop contrasts of diffusion vs clamped BC whose verdict equals the frozen "
                 "contrast's (same family without the clamp), both tasks",
    "x1.kept.of": "number of such drop contrasts",
    "x3.nevals": "number of X3 evaluations listed in results.json",
    "x3.newseeds": "the seeds beyond the frozen three (first and last)",
    "x2.nevals": "number of X2 evaluations listed in results.json",
    "x2.horizon": "parsed from results.json design.outcome and checked against the censored-"
                  "length rule",
    "*.perseed": "per-seed rates joined with commas, at the Level 1 tables' whole-percent rounding",
}


def _verdict(c: dict[str, Any]) -> str:
    v = c["verdict"]
    return fmt.sep(f"{v} ({c['sign']})" if c.get("sign") else v)


def _diff(n: Numbers, key: str, d: float, ci: list[float], sep: str) -> None:
    n.set(f"{key}.d", fmt.spct(d))
    n.set(f"{key}.lo", fmt.spct(ci[0]))
    n.set(f"{key}.hi", fmt.spct(ci[1]))
    n.set(f"{key}.sep", fmt.sep(sep))


def _family(n: Numbers, key: str, v: dict[str, Any]) -> None:
    n.set(f"{key}.succ", fmt.pct(v["estimate"]))
    n.set(f"{key}.lo", fmt.pct(v["ci95"][0]))
    n.set(f"{key}.hi", fmt.pct(v["ci95"][1]))
    n.set(f"{key}.perseed", ", ".join(fmt.pct0(x) for x in v["per_seed"]))
    n.set(f"{key}.nseeds", v["n_seeds"])
    n.set(f"{key}.nstates", v["n_states"])


def _contrast(n: Numbers, key: str, c: dict[str, Any]) -> None:
    _diff(n, f"{key}.b", c["difference"], c["bootstrap_ci95"], c["bootstrap_separation"])
    n.set(f"{key}.s.lo", fmt.spct(c["seed_level_ci95"][0]))
    n.set(f"{key}.s.hi", fmt.spct(c["seed_level_ci95"][1]))
    n.set(f"{key}.s.sep", fmt.sep(c["seed_level_separation"]))
    n.set(f"{key}.verdict", _verdict(c))
    n.set(f"{key}.relation", c["relation"])


def _frozen_contrast(n: Numbers, key: str, f: dict[str, Any]) -> None:
    _diff(n, f"{key}.b", f["difference"], f["bootstrap_ci95"], f["bootstrap_separation"])
    n.set(f"{key}.s.lo", fmt.spct(f["seed_level_ci95"][0]))
    n.set(f"{key}.s.hi", fmt.spct(f["seed_level_ci95"][1]))
    n.set(f"{key}.s.sep", fmt.sep(f["seed_level_separation"]))


def _by_name(contrasts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {c["name"]: c for c in contrasts}
    missing = sorted(set(CONTRASTS.values()) - set(out))
    if missing:
        raise KeyError(f"contrasts missing from the results: {missing}")
    return out


def x1_numbers(n: Numbers, x1: dict[str, Any]) -> None:
    """X1: the preregistered primary, its supporting and base legs, success and clamp activity."""
    r = x1["readings"]
    _diff(n, "x1.prim", r["primary"]["estimate"], r["primary"]["ci95"],
          r["primary"]["separation"])
    f = r["primary"]["frozen"]
    _diff(n, "x1.prim.f", f["estimate"], f["ci95"], f["separation"])
    _diff(n, "x1.sup", r["support"]["estimate"], r["support"]["ci95"],
          r["support"]["separation"])
    for task, b in r["base"].items():
        _diff(n, f"x1.base.{task}", b["difference"], b["ci95"], b["separation"])
    n.set("x1.nevals", sum(len(t["evaluations"]) for t in x1["tasks"].values()))
    n.set("x1.B", fmt.thousands(x1["tasks"]["lift"]["bootstrap"]["B"]))
    cu, kept, of = [], 0, 0
    for task, t in x1["tasks"].items():
        n.set(f"x1.{task}.nstates", t["n_states"])
        bs = t["bootstrap"]
        for ak, arm in X1_ARMS.items():
            for s, sk in SETS.items():
                v = bs["families"].get(f"{arm}:{s}")
                if v is not None:
                    _family(n, f"x1.{task}.{ak}.{sk}", v)
        cu += [d for d in bs["x1_differences"] if d["kind"].startswith("clamped - unclamped")]
        frozen = {c["name"]: c for c in bs["frozen_contrasts"]}
        for c in bs["x1_contrasts"]:
            m = re.fullmatch(r"drop diffusion_unet - drop (\S+)\+clamp on (\S+)", c["name"])
            if m is None:
                continue
            fr = frozen[f"drop diffusion_unet - drop {m.group(1)} on {m.group(2)}"]
            of += 1
            kept += c["separation"] == fr["separation"]
            fam = {"bc_mlp": "bcmlpc", "bc_rnn": "bcrnnc", "bc_mlp_chunk": "chunkc"}[m.group(1)]
            _diff(n, f"x1.{task}.dd.{fam}.{SETS[m.group(2)]}", c["estimate"], c["ci95"],
                  c["separation"])
            _diff(n, f"x1.{task}.dd.{fam.rstrip('c')}.{SETS[m.group(2)]}.f", fr["estimate"],
                  fr["ci95"], fr["separation"])
        for arm_set, a in t["clamp_activity"].items():
            arm, s = arm_set.split(":")
            ak = {v: k for k, v in X1_ARMS.items()}[arm]
            n.set(f"x1.{task}.chg.{ak}.{SETS[s]}", fmt.pct(a["fraction_actions_changed"]))
    n.set("x1.cu.n", len(cu))
    n.set("x1.cu.nsep", sum(d["separation"] != "not separated" for d in cu))
    n.set("x1.cu.maxabs", fmt.pct(max(abs(d["difference"]) for d in cu)))
    n.set("x1.kept.n", kept)
    n.set("x1.kept.of", of)


def x3_numbers(n: Numbers, x3: dict[str, Any]) -> None:
    """X3: the four declared Can contrasts beside the frozen ones, and family success."""
    p = x3["primary"]
    cs = _by_name(p["contrasts"])
    for k, name in CONTRASTS.items():
        _contrast(n, f"x3.{k}", cs[name])
        _frozen_contrast(n, f"l1.can.{k}", cs[name]["frozen"])
    for fam in FAMILIES:
        fk = fmt.FAMILY_KEY[fam]
        _family(n, f"x3.fam.{fk}", p["bootstrap"]["families"][fam])
        _family(n, f"l1.can.fam.{fk}", x3["frozen_families"][fam])
    frozen_seeds = x3["frozen_families"]["diffusion_unet"]["n_seeds"]
    new = [s for s in p["seeds"] if s >= frozen_seeds]
    n.set("x3.newseeds", f"{new[0]}–{new[-1]}")
    n.set("x3.nnew", len(new))
    n.set("x3.nseeds", len(p["seeds"]))
    n.set("x3.nstates", x3["suite"]["n_states"])
    n.set("x3.nevals", len(x3["evaluations"]))
    n.set("x3.B", fmt.thousands(p["bootstrap"]["B"]))
    n.set("x3.errors", sum(e.get("n_errors", 0) for e in x3["evaluations"]))


def _x2_horizon(x2: dict[str, Any]) -> int:
    a = re.search(r"before the (\d+)-step horizon", x2["design"]["outcome"])
    b = re.search(r"else (\d+)", x2["design"]["secondary"]["censored_episode_length"])
    if a is None or b is None or a.group(1) != b.group(1):
        raise ValueError("X2 horizon not stated consistently in results.json design")
    return int(a.group(1))


def x2_numbers(n: Numbers, x2: dict[str, Any]) -> None:
    """X2 (Can MH): contrasts with their relation to Level 1 Can PH, success, censored length."""
    p = x2["primary"]
    cs = _by_name(p["contrasts"])
    horizons = {c["frozen"]["horizon"] for c in cs.values()}
    if len(horizons) != 1:
        raise ValueError("the frozen Level 1 horizon differs between contrasts")
    for k, name in CONTRASTS.items():
        _contrast(n, f"x2.{k}", cs[name])
        _frozen_contrast(n, f"l1.x2side.{k}", cs[name]["frozen"])
    cens = {(d["a"], d["b"]): d for d in p["censored_episode_length"]["differences"]}
    for k, name in CONTRASTS.items():
        a, b = cs[name]["a"], cs[name]["b"]
        d = cens[(a, b)]
        n.set(f"x2.cens.{k}.d", fmt.snum1(d["difference"]))
        n.set(f"x2.cens.{k}.lo", fmt.snum1(d["ci95"][0]))
        n.set(f"x2.cens.{k}.hi", fmt.snum1(d["ci95"][1]))
        n.set(f"x2.cens.{k}.sep", fmt.sep(d["separation"]))
    for fam in FAMILIES:
        fk = fmt.FAMILY_KEY[fam]
        _family(n, f"x2.fam.{fk}", p["bootstrap"]["families"][fam])
        _family(n, f"l1.x2side.fam.{fk}", x2["frozen_families"][fam])
        for cat, cnt in x2["taxonomy"][fam].items():
            n.set(f"x2.tax.{fk}.{cat.replace('_', '-')}", cnt)
    n.set("x2.horizon", _x2_horizon(x2))
    n.set("l1.horizon", horizons.pop())
    n.set("x2.nstates", x2["suite"]["n_states"])
    n.set("x2.nseeds", p["bootstrap"]["families"]["diffusion_unet"]["n_seeds"])
    n.set("x2.nevals", len(x2["evaluations"]))
    n.set("x2.B", fmt.thousands(p["bootstrap"]["B"]))
    n.set("x2.errors", x2["engineering"]["n_errors_total"])
    n.set("x2.attempt2", x2["engineering"]["n_attempt2"])
    levels = {x2["primary"]["bootstrap"]["level"], x2["design"]["bootstrap"]["level"]}
    if len(levels) != 1:
        raise ValueError("X2 bootstrap level stated inconsistently")
    n.set("level", fmt.pct0(levels.pop()))


def build_numbers(sources: dict[str, dict[str, Any]]) -> Numbers:
    """All addendum macros from the three pinned results (see :func:`sources.load_all`)."""
    n = Numbers()
    x1_numbers(n, sources["x1"])
    x3_numbers(n, sources["x3"])
    x2_numbers(n, sources["x2"])
    return n
