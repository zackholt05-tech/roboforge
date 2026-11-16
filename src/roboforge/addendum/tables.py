"""LaTeX tables of the v1.1 addendum, every cell an ``\\ax{key}`` macro (D-133).

The tables hold no literal number: each cell names a generated macro, so ``numbers.tex`` is the
only place a value lives. Column heads state each study's label, design and horizon, so X2 (Can
MH) and the frozen Level 1 Can PH result stand beside each other and are never pooled.
"""

from __future__ import annotations

from collections.abc import Callable

from roboforge.addendum.numbers import CONTRASTS
from roboforge.addendum.sources import SOURCES

LABEL = {
    "h2bcmlp": "H2 diffusion $-$ BC-MLP",
    "h2bcrnn": "H2 diffusion $-$ BC-RNN",
    "h2c": "H2-c diffusion $-$ BC-MLP-chunk",
    "h3": "H3 BC-RNN $-$ BC-MLP",
}
FAMS = (("diff", "Diffusion U-Net"), ("chunk", "BC-MLP-chunk"), ("bcrnn", "BC-RNN"),
        ("bcmlp", "BC-MLP"))


def ax(key: str) -> str:
    """A macro reference."""
    return f"\\ax{{{key}}}"


def diff(key: str) -> str:
    """``d [lo, hi]`` of a difference macro family."""
    return f"{ax(key + '.d')}\\,[{ax(key + '.lo')}, {ax(key + '.hi')}]"


def _ci(key: str) -> str:
    return f"[{ax(key + '.lo')}, {ax(key + '.hi')}]"


def x1_contrasts() -> str:
    """X1's preregistered primary, supporting and base legs beside the frozen values."""
    rows = [
        ("Primary: Lift arm offset, drop(diffusion) $-$ drop(BC-MLP-chunk)",
         diff("x1.prim"), ax("x1.prim.sep"), diff("x1.prim.f"), ax("x1.prim.f.sep")),
        ("Supporting: drop(BC-MLP-chunk) $-$ drop(clamped), Lift arm offset",
         diff("x1.sup"), ax("x1.sup.sep"), "", ""),
        ("Base leg, Lift: clamped $-$ unclamped BC-MLP-chunk", diff("x1.base.lift"),
         ax("x1.base.lift.sep"), "", ""),
        ("Base leg, Can: clamped $-$ unclamped BC-MLP-chunk", diff("x1.base.can"),
         ax("x1.base.can.sep"), "", ""),
    ]
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (
        "\\begin{tabular}{p{4.4cm}llll}\n\\toprule\n"
        "Difference (pp, " + ax("level") + "\\,\\% interval) & X1, clamped & verdict & "
        "frozen, unclamped & verdict \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def x1_success() -> str:
    """Lift success by arm and set (Session 7 robustness sets), and how often the clamp acted."""
    arms = (("diff", "Diffusion U-Net (sampler bound)"), ("chunk", "BC-MLP-chunk"),
            ("chunkc", "BC-MLP-chunk, clamped"), ("bcmlp", "BC-MLP"),
            ("bcmlpc", "BC-MLP, clamped"), ("bcrnn", "BC-RNN"), ("bcrnnc", "BC-RNN, clamped"))
    rows = []
    for k, label in arms:
        cells = [f"{ax(f'x1.lift.{k}.{s}.succ')} {_ci(f'x1.lift.{k}.{s}')}"
                 for s in ("base", "cube", "arm")]
        chg = ax(f"x1.lift.chg.{k}.arm") if k.endswith("c") else "--"
        rows.append(" & ".join([label, *cells, chg]) + r" \\")
    return (
        "\\begin{tabular}{lllll}\n\\toprule\n"
        "Arm & base & cube shift & arm offset & changed (\\%) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def _contrast_table(study: str, frozen: str, name: str, relation: bool) -> str:
    """Two rows per contrast: the frozen Level 1 Can PH value, then the extension's."""
    rows = []
    for k in CONTRASTS:
        f, s = f"{frozen}.{k}", f"{study}.{k}"
        verdict = ax(s + ".verdict") + (f"; {ax(s + '.relation')}" if relation else "")
        rows.append(f"{LABEL[k]} & Level~1 Can PH & {diff(f + '.b')} & {_ci(f + '.s')} & "
                    f"{ax(f + '.b.sep')} \\\\")
        rows.append(f" & {name} & {diff(s + '.b')} & {_ci(s + '.s')} & {verdict} \\\\[2pt]")
    head = ("Contrast & study & difference, bootstrap & seed-level & verdict \\\\\n")
    return ("\\begin{tabular}{llllp{3.4cm}}\n\\toprule\n" + head + "\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def x3_contrasts() -> str:
    """X3's four declared Can contrasts below the frozen Level 1 Can PH values."""
    return _contrast_table("x3", "l1.can", "X3", relation=False)


def x2_contrasts() -> str:
    """X2 (Can MH)'s four contrasts below the frozen Level 1 Can PH values, with the relation."""
    return _contrast_table("x2", "l1.x2side", "X2 (Can MH)", relation=True)


def families() -> str:
    """Family success: frozen Level 1 Can PH, X3 and X2 (Can MH) in separate columns."""
    rows = []
    for k, label in FAMS:
        cells = [label]
        for p in (f"l1.can.fam.{k}", f"x3.fam.{k}", f"x2.fam.{k}"):
            cells.append(f"{ax(p + '.succ')} {_ci(p)}")
        cells.append(ax(f"x2.fam.{k}.perseed"))
        rows.append(" & ".join(cells) + r" \\")
    head = ("Family & Level~1 Can PH (frozen) & X3 & X2 (Can MH) & X2 per seed \\\\\n"
            f" & {ax('l1.can.fam.diff.nseeds')} seeds $\\times$ {ax('l1.can.fam.diff.nstates')},"
            f" {ax('l1.horizon')} steps & {ax('x3.nseeds')} seeds $\\times$ {ax('x3.nstates')},"
            f" {ax('l1.horizon')} steps & {ax('x2.nseeds')} seeds $\\times$ {ax('x2.nstates')},"
            f" {ax('x2.horizon')} steps & \\% \\\\\n")
    return ("\\begin{tabular}{lllll}\n\\toprule\n" + head + "\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def x2_censored() -> str:
    """X2's declared secondary: censored episode length (steps), no readings."""
    rows = [f"{LABEL[k]} & {diff(f'x2.cens.{k}')} & {ax(f'x2.cens.{k}.sep')} \\\\"
            for k in CONTRASTS]
    return ("\\begin{tabular}{lll}\n\\toprule\nContrast & difference (steps, "
            + ax("level") + "\\,\\%) & verdict \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def source_files() -> str:
    """The three pinned results with their SHA-256 (from the source pins)."""
    rows = [f"\\file{{{s.path}}} & \\code{{{s.sha256}}} \\\\" for s in SOURCES.values()]
    return ("{\\scriptsize\\begin{tabular}{ll}\n\\toprule\nFile & SHA-256 \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")


TABLES: dict[str, Callable[[], str]] = {
    "x1_contrasts": x1_contrasts,
    "x1_success": x1_success,
    "x3_contrasts": x3_contrasts,
    "x2_contrasts": x2_contrasts,
    "families": families,
    "x2_censored": x2_censored,
    "sources": source_files,
}
