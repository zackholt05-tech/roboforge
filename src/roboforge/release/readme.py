"""The README's key-findings block, generated from the paper's frozen-result macros (D-103).

The block sits between two marker comments in `README.md`. `scripts/readme_findings.py`
rewrites it; `check_readme` (run by `make release-check`) fails if the committed block differs
from a fresh generation. Every number comes from `paper/generated/numbers.tex` through
`Macros`, so an unknown key fails the build instead of printing a guess. The prose around the
numbers states the audit's caveats (KI-052, KI-053, KI-034) next to the results they qualify.
"""

from __future__ import annotations

from pathlib import Path

from roboforge.paths import REPO_ROOT
from roboforge.release.macros import NUMBERS_TEX, Macros, load_macros

README = REPO_ROOT / "README.md"
BEGIN = (
    "<!-- BEGIN GENERATED: findings (scripts/readme_findings.py from "
    "paper/generated/numbers.tex; do not edit by hand) -->"
)
END = "<!-- END GENERATED: findings -->"


def _families(m: Macros) -> tuple[tuple[str, str], ...]:
    """(macro family key, row label) for the Level 1 held-out table, in the paper's order."""
    return (
        ("bcmlp", "BC-MLP (one step)"),
        ("bcrnn", "BC-RNN (LSTM)"),
        ("chunk", f"BC-MLP-chunk ({m['hor.pred']}-step chunks)"),
        ("diff", f"Diffusion U-Net ({m['diff.ddim']} DDIM steps)"),
        ("random", "Random actions (sanity)"),
    )


def _cell(m: Macros, task: str, fam: str) -> str:
    p = f"l1.{task}.{fam}"
    rate = f"**{m[p + '.succ']} %** ({m[p + '.k']}/{m[p + '.n']})"
    if fam == "random":
        return rate
    return f"{rate} [{m[p + '.lo']}, {m[p + '.hi']}]"


def _diff(m: Macros, key: str) -> str:
    """``d pp [lo, hi], verdict`` for a difference macro family."""
    return f"{m[key + '.d']} pp [{m[key + '.lo']}, {m[key + '.hi']}], {m[key + '.sep']}"


def _need(m: Macros, key: str, *allowed: str) -> None:
    """The prose states a verdict; refuse to generate it if the frozen macro says otherwise."""
    if m[key] not in allowed:
        raise ValueError(f"{key} is {m[key]!r}; the README text assumes one of {allowed}")


SEP_POS, SEP_NEG, NOT_SEP = "separated (+)", "separated (−)", "not separated"


def findings_block(m: Macros) -> str:
    """Return the Markdown between (and including) the BEGIN and END markers."""
    # Every qualitative statement in the text below is checked against the frozen verdicts.
    _need(m, "l1.can.diff-bcmlp.sep", SEP_POS)
    _need(m, "sens.l1.can.diff-bcmlp.sep", NOT_SEP)
    _need(m, "l1.lift.diff-bcmlp.sep", NOT_SEP)
    for task in ("lift", "can"):
        _need(m, f"l1.{task}.diff-chunk.sep", NOT_SEP)
        _need(m, f"l1.{task}.diff-bcrnn.sep", NOT_SEP)
        _need(m, f"l1.{task}.bcrnn-bcmlp.sep", SEP_POS)
    _need(m, "sens.l1.can.bcrnn-bcmlp.sep", NOT_SEP)
    for other in ("bcmlp", "bcrnn", "chunk"):
        _need(m, f"rob.lift.dd.arm.diff-{other}.sep", SEP_NEG)
    _need(m, "rob.can.dd.arm.diff-chunk.sep", NOT_SEP)
    _need(m, "rgb.diff-bc.sep", NOT_SEP)
    _need(m, "l2.h5.sep", SEP_NEG)
    n_ep = int(m["proto.nseeds"]) * int(m["proto.nstates"])
    rows = "\n".join(
        f"| {label} | {_cell(m, 'lift', fam)} | {_cell(m, 'can', fam)} |"
        for fam, label in _families(m)
    )
    lines = [
        BEGIN,
        "",
        f"**Level 1 held-out success** (exact simulator state as input, which is privileged "
        f"information; {m['proto.nseeds']} training seeds × {m['proto.nstates']} paired held-out "
        f"starts = {n_ep} episodes per cell, run once under the frozen protocol; "
        f"{m['proto.level']} % hierarchical bootstrap interval, preregistered):",
        "",
        "| Policy | Lift | Can |",
        "|---|---|---|",
        rows,
        "",
        "Differences below are in percentage points (pp) with the preregistered "
        f"{m['proto.level']} % interval and verdict.",
        "",
        "- **Diffusion − BC-MLP (H2):** Can "
        + _diff(m, "l1.can.diff-bcmlp")
        + ". This separation is fragile: a seed-level t interval, computed after the audit and "
        f"not preregistered, is [{m['sens.l1.can.diff-bcmlp.lo']}, "
        f"{m['sens.l1.can.diff-bcmlp.hi']}] and includes 0 (KI-053). Lift "
        + _diff(m, "l1.lift.diff-bcmlp")
        + f" (seed-level interval [{m['sens.l1.lift.diff-bcmlp.lo']}, "
        f"{m['sens.l1.lift.diff-bcmlp.hi']}], {m['sens.l1.lift.diff-bcmlp.sep']}).",
        "- **Diffusion − BC-MLP-chunk (H2-c, same inputs, horizons and chunking; deterministic "
        "head):** Lift "
        + _diff(m, "l1.lift.diff-chunk")
        + "; Can "
        + _diff(m, "l1.can.diff-chunk")
        + ". These data give no evidence that the generative model adds to observation history "
        "and action chunking. Diffusion is also not separated from BC-RNN on either task.",
        "- **Action bound (KI-052):** the diffusion sampler keeps every action inside the "
        "training range; the BC heads are bounded only by the controller limits. This confounds "
        "H2-c and the robustness comparisons.",
        "- **BC-RNN − BC-MLP (H3):** Lift "
        + _diff(m, "l1.lift.bcrnn-bcmlp")
        + "; Can "
        + _diff(m, "l1.can.bcrnn-bcmlp")
        + f" (seed-level interval [{m['sens.l1.can.bcrnn-bcmlp.lo']}, "
        f"{m['sens.l1.can.bcrnn-bcmlp.hi']}], fragile in the same way).",
        f"- **Cost:** diffusion took {m['l1.ratio.cpu.lo']}–{m['l1.ratio.cpu.hi']}× the chunked "
        f"MLP's CPU training time, and a diffusion decision ({m['diff.ddim']} DDIM steps) took "
        f"{m['l1.ratio.lat.lo']}–{m['l1.ratio.lat.hi']}× as long.",
        "- **Robustness (H4, preregistered perturbation sets, not the held-out test):** under "
        "the initial arm offset, diffusion degrades least on Lift (its drop is separated from "
        "each BC family's; minus the chunked MLP's: "
        + _diff(m, "rob.lift.dd.arm.diff-chunk")
        + "); on Can the same contrast is "
        + _diff(m, "rob.can.dd.arm.diff-chunk")
        + ". KI-052 applies to both.",
        f"- **RGB study (Lift; two {m['vis.image']}×{m['vis.image']} cameras plus "
        "proprioception, ResNet-18 encoder):** visual "
        f"diffusion {m['rgb.rgbdiff.succ']} % ({m['rgb.rgbdiff.k']}/{m['rgb.rgbdiff.n']}) vs "
        f"visual BC-chunk {m['rgb.rgbbc.succ']} % ({m['rgb.rgbbc.k']}/{m['rgb.rgbbc.n']}): "
        + _diff(m, "rgb.diff-bc")
        + ". Lift saturates for every policy, so it cannot rank them (KI-034).",
        "- **Level 2 (Lift; DP3-inspired point clouds from sensor-like depth, "
        f"{m['pc.points']} points):** under cube placements outside the demonstrations' range, "
        f"the point-cloud diffusion policy reached {m['l2.pcdiff.cube.succ']} % vs "
        f"{m['l2.rgbdiff.cube.succ']} % for the frozen RGB diffusion policy (same head and "
        "training, different encoder and augmentation): " + _diff(m, "l2.h5") + ". H5 is "
        "not supported; the explicit 3D input generalized worse here.",
        f"- **Paid compute:** ${m['spend.total']} over {m['spend.launches']} Modal launches "
        "(list price before credits); Level 1 trained on a local CPU.",
        "",
        END,
    ]
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    """Return ``text`` with the marked block replaced; exactly one block must exist."""
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("README must contain exactly one findings BEGIN and END marker")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return head + block + tail


def check_readme(readme: Path = README, numbers: Path = NUMBERS_TEX) -> list[str]:
    """Problems with the committed README block (empty when it equals a fresh generation)."""
    text = readme.read_text(encoding="utf-8")
    try:
        fresh = replace_block(text, findings_block(Macros(load_macros(numbers))))
    except (ValueError, KeyError) as exc:
        return [f"README findings block: {exc}"]
    if fresh != text:
        return ["README findings block differs from scripts/readme_findings.py output "
                "(run make readme-findings)"]
    return []
