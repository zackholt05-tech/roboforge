"""Seed-level sensitivity intervals, labelled not preregistered (DECISIONS D-095).

The frozen analyses use the preregistered hierarchical paired bootstrap. With 3 training
seeds per family its seed component rests on 2 degrees of freedom, so the audit (B-1) asked how
much each verdict depends on the method. This module recomputes every declared difference from
the frozen ``per_seed`` values only:

- families trained separately: Welch t on the per-seed rates (Welch–Satterthwaite df);
- the same checkpoints under two conditions (an OOD drop, A1, A2): paired t on the per-seed
  differences;
- a difference of drops: Welch on the per-seed drops;
- a non-learned baseline with one policy seed is a constant (one-sample t on the other side).

It treats per-seed means as the units and ignores pairing by start, so it checks the seed
dimension only. It never replaces a frozen verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from scipy import stats

from roboforge.paper import fmt
from roboforge.paper.sources import Sources

LEVEL = 0.95
TASKS = ["lift", "can"]


@dataclass
class Row:
    """One declared difference: the frozen verdict and the seed-level t interval."""

    analysis: str
    name: str
    key: str
    frozen: float
    frozen_ci: tuple[float, float]
    frozen_sep: str
    method: str
    ci: tuple[float, float] | None
    df: float | None

    @property
    def sep(self) -> str:
        """Separation of the sensitivity interval, with the frozen rule."""
        if self.ci is None:
            return "undefined (no seed variance)"
        lo, hi = self.ci
        return "separated (+)" if lo > 0 else "separated (-)" if hi < 0 else "not separated"

    @property
    def changed(self) -> bool:
        """True when the sensitivity verdict differs from the frozen one."""
        return self.ci is not None and self.sep != self.frozen_sep


def _var(x: list[float]) -> float:
    m = sum(x) / len(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def welch(a: list[float], b: list[float]) -> tuple[tuple[float, float] | None, float | None]:
    """Welch t interval for mean(a) - mean(b); one side of size 1 is a constant."""
    d = sum(a) / len(a) - sum(b) / len(b)
    parts = [(_var(x) / len(x), len(x) - 1) for x in (a, b) if len(x) > 1]
    se2 = sum(p for p, _ in parts)
    if se2 == 0:
        return None, None
    df = se2 ** 2 / sum(p ** 2 / k for p, k in parts)
    h = stats.t.ppf(0.5 + LEVEL / 2, df) * math.sqrt(se2)
    return (d - h, d + h), df


def paired(a: list[float], b: list[float]) -> tuple[tuple[float, float] | None, float | None]:
    """Paired t interval for the mean per-seed difference a_i - b_i."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    v = _var(diffs)
    if v == 0:
        return None, None
    df = len(diffs) - 1
    h = stats.t.ppf(0.5 + LEVEL / 2, df) * math.sqrt(v / len(diffs))
    m = sum(diffs) / len(diffs)
    return (m - h, m + h), df


def _row(analysis: str, name: str, key: str, d: dict[str, Any], method: str,
         res: tuple[tuple[float, float] | None, float | None], value_key: str = "difference"
         ) -> Row:
    return Row(analysis, name, key, d[value_key], tuple(d["ci95"]), d["separation"], method,
               res[0], res[1])


def _lab(f: str) -> str:
    return fmt.FAMILY_LABEL[f].replace("Diffusion U-Net", "Diffusion")


def rows(src: Sources) -> list[Row]:
    """Every declared difference with its frozen and seed-level intervals."""
    out: list[Row] = []
    r = src.json("l1_results")
    for t in TASKS:
        fams = r["tasks"][t]["success"]["families"]
        for d in r["tasks"][t]["success"]["differences"]:
            a, b = fams[d["a"]]["per_seed"], fams[d["b"]]["per_seed"]
            out.append(_row(f"Level 1 held-out, {t.capitalize()}",
                            f"{d['hypothesis'].replace('H2c', 'H2-c')}: {_lab(d['a'])} $-$ "
                            f"{_lab(d['b'])}",
                            f"l1.{t}.{fmt.FAMILY_KEY[d['a']]}-{fmt.FAMILY_KEY[d['b']]}", d,
                            "one-sample t" if len(b) == 1 else "Welch t", welch(a, b)))
    rgb = src.json("rgb_results")["success"]
    d = rgb["differences"][0]
    out.append(_row("RGB study held-out, Lift", f"{_lab(d['a'])} $-$ {_lab(d['b'])}",
                    "rgb.diff-bc", d, "Welch t", welch(rgb["families"][d["a"]]["per_seed"],
                                                       rgb["families"][d["b"]]["per_seed"])))
    l2 = src.json("l2_results")
    pb = l2["placement"]["bootstrap"]
    for d in pb["differences"]:
        a, b = pb["families"][d["a"]]["per_seed"], pb["families"][d["b"]]["per_seed"]
        fa, sa = d["a"].split(":")
        fb, _ = d["b"].split(":")
        if fa == fb:
            name, key, res, m = (f"drop of {_lab(fa)} (base $-$ cube shift)",
                                 f"l2.{fmt.FAMILY_KEY[fa]}.drop", paired(a, b), "paired t")
        else:
            prim = "H5: " if d["kind"].startswith("PRIMARY") else ""
            name = f"{prim}point cloud $-$ RGB on {fmt.SET_LABEL[sa]}"
            key = "l2.h5" if prim else "l2.base"
            res, m = welch(a, b), "Welch t"
        out.append(_row("Level 2 placement, Lift", name, key, d, m, res))
    c = pb["contrasts"][0]
    drops = {}
    for arm in ("pc_diffusion", "rgb_diffusion"):
        base = pb["families"][f"{arm}:robustness_base"]["per_seed"]
        ood = pb["families"][f"{arm}:ood_cube_shift"]["per_seed"]
        drops[arm] = [x - y for x, y in zip(base, ood, strict=True)]
    out.append(_row("Level 2 placement, Lift", "drop point cloud $-$ drop RGB", "l2.dd", c,
                    "Welch t", welch(drops["pc_diffusion"], drops["rgb_diffusion"]),
                    value_key="estimate"))
    hs = l2["heldout"]["success"]
    d = hs["differences"][0]
    out.append(_row("Level 2 held-out, Lift", "point cloud $-$ RGB", "l2.held", d, "Welch t",
                    welch(hs["families"][d["a"]]["per_seed"], hs["families"][d["b"]]["per_seed"])))
    out += robustness_rows(src)
    return out


def _drop(fams: dict[str, Any], f: str, st: str) -> list[float]:
    """Per-seed drop base - ``st`` of family ``f``."""
    base = fams[f"{f}:robustness_base"]["per_seed"]
    ood = fams[f"{f}:{st}"]["per_seed"]
    return [x - y for x, y in zip(base, ood, strict=True)]


def robustness_rows(src: Sources) -> list[Row]:
    """H4 and A1-A3 (validation regime) with seed-level intervals."""
    a = src.json("l1_robustness")
    out: list[Row] = []
    for t in TASKS:
        b = a["tasks"][t]["h4"]["bootstrap"]
        fams = b["families"]
        an = f"H4, {t.capitalize()}"
        for d in b["differences"]:
            fa, sa = d["a"].split(":")
            fb, sb = d["b"].split(":")
            pa, pb_ = fams[d["a"]]["per_seed"], fams[d["b"]]["per_seed"]
            if fa == fb:
                out.append(_row(an, f"drop of {_lab(fa)}, {fmt.SET_LABEL[sb]}",
                                f"rob.{t}.{fmt.FAMILY_KEY[fa]}.drop.{fmt.SET_KEY[sb]}", d,
                                "paired t", paired(pa, pb_)))
            else:
                out.append(_row(an, f"{_lab(fa)} $-$ {_lab(fb)} on {fmt.SET_LABEL[sa]}",
                                f"rob.{t}.{fmt.SET_KEY[sa]}.{fmt.FAMILY_KEY[fa]}-"
                                f"{fmt.FAMILY_KEY[fb]}", d, "Welch t", welch(pa, pb_)))
        for c in b["contrasts"]:
            pos = [k for k, w in c["weights"].items() if w == 1]
            fa = next(k.split(":")[0] for k in pos if k.endswith("robustness_base"))
            fb = next(k.split(":")[0] for k in pos if not k.endswith("robustness_base"))
            st = next(k.split(":")[1] for k in pos if not k.endswith("robustness_base"))

            out.append(_row(an, f"drop {_lab(fa)} $-$ drop {_lab(fb)}, {fmt.SET_LABEL[st]}",
                            f"rob.{t}.dd.{fmt.SET_KEY[st]}.{fmt.FAMILY_KEY[fa]}-"
                            f"{fmt.FAMILY_KEY[fb]}", c, "Welch t",
                            welch(_drop(fams, fa, st), _drop(fams, fb, st)),
                            value_key="estimate"))
        for ab in ("A1", "A2", "A3"):
            blk = a["tasks"][t].get(ab)
            if blk is None:
                continue
            d = blk["bootstrap"]["differences"][0]
            pa = blk["bootstrap"]["families"][d["a"]]["per_seed"]
            pb_ = blk["bootstrap"]["families"][d["b"]]["per_seed"]
            res, m = (welch(pa, pb_), "Welch t") if ab == "A3" else (paired(pa, pb_), "paired t")
            out.append(_row(f"{ab}, {t.capitalize()}", f"{d['a']} $-$ {d['b']}".replace(
                "_", "\\_"), f"abl.{t}.{ab}", d, m, res))
    return out


def table(rs: list[Row]) -> str:
    """LaTeX tabular: frozen verdict beside the seed-level sensitivity verdict."""
    lines = ["\\begin{tabular}{llll}", "\\toprule",
             "Comparison & Frozen bootstrap (pp) & Seed-level $t$ (pp, df) & \\\\", "\\midrule"]
    current = None
    for r in rs:
        if r.analysis != current:
            if current is not None:
                lines.append("\\addlinespace")
            lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{r.analysis}}}}} \\\\")
            current = r.analysis
        frozen = f"{fmt.spct(r.frozen)} {fmt.sci(*r.frozen_ci)}, {fmt.sep(r.frozen_sep)}"
        if r.ci is None:
            sens = "undefined (no seed variance)"
        else:
            sens = f"{fmt.sci(*r.ci)} ({r.df:.1f}), {fmt.sep(r.sep)}"
        mark = "$\\ast$" if r.changed else ""
        lines.append(f"{r.name} & {frozen} & {sens} & {mark} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def numbers(rs: list[Row], n: Any) -> None:
    """``sens.*`` macros: each interval, and how many verdicts change per study."""
    for r in rs:
        if r.ci is not None:
            n.set(f"sens.{r.key}.lo", fmt.spct(r.ci[0]))
            n.set(f"sens.{r.key}.hi", fmt.spct(r.ci[1]))
            n.set(f"sens.{r.key}.df", f"{r.df:.1f}")
        n.set(f"sens.{r.key}.sep", fmt.sep(r.sep))
    groups = {"l1": lambda r: r.analysis.startswith("Level 1"),
              "rob": lambda r: r.analysis.startswith(("H4", "A1", "A2", "A3")),
              "ext": lambda r: r.analysis.startswith(("RGB", "Level 2"))}
    for g, sel in groups.items():
        rr = [r for r in rs if sel(r)]
        n.set(f"sens.{g}.n", len(rr))
        n.set(f"sens.{g}.changed", sum(r.changed for r in rr))
        n.set(f"sens.{g}.undefined", sum(r.ci is None for r in rr))
        lost = [r for r in rr if r.changed and r.frozen_sep != "not separated"]
        gained = [r for r in rr if r.changed and r.frozen_sep == "not separated"]
        n.set(f"sens.{g}.lost", len(lost))
        n.set(f"sens.{g}.gained", len(gained))
    n.set("sens.all.n", len(rs))
    # Audit B-4: frozen intervals with a bound at exactly zero or zero width, where the
    # verdict rests on ties of a discrete bootstrap distribution.
    n.set("sens.all.boundary", sum(r.frozen_ci[0] == 0 or r.frozen_ci[1] == 0
                                   or r.frozen_ci[0] == r.frozen_ci[1] for r in rs))
