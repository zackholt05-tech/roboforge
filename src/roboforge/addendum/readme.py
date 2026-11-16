"""The README's ``extensions`` block, generated from the addendum's macros (DECISIONS D-133).

The block sits between its own markers in ``README.md``, after the v1.0 findings block, which
stays as generated from ``paper/generated/numbers.tex``. Every number comes from
``paper/addendum/generated/numbers.tex`` (which ``make addendum-check`` proves equal to a fresh
build from the three pinned results). Every qualitative statement is checked against the
generated verdicts before it is written, so a changed verdict stops the generator instead of
leaving stale prose. ``check`` (run by ``make release-check``) fails if the committed block or
the chart differs from a fresh generation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from roboforge.addendum.build import CHART, OUT, PREFIX
from roboforge.paths import REPO_ROOT
from roboforge.release.macros import Macros, to_text

README = "README.md"
NUMBERS = f"{OUT}/numbers.tex"
NAME = "extensions"
BEGIN = (f"<!-- BEGIN GENERATED: {NAME} (scripts/readme_extensions.py from {NUMBERS}; "
         "do not edit by hand) -->")
END = f"<!-- END GENERATED: {NAME} -->"
_LINE = re.compile(r"^\\expandafter\\def\\csname " + re.escape(PREFIX)
                   + r"(?P<key>[^\\]+)\\endcsname\{(?P<val>.*)\}$")
ROBUST, NOT_SEP = "robust (+)", "not separated"


def load(root: Path = REPO_ROOT) -> Macros:
    """The addendum macros as plain text (strict accessor)."""
    out: dict[str, str] = {}
    for line in (root / NUMBERS).read_text(encoding="utf-8").splitlines():
        m = _LINE.match(line.strip())
        if m:
            if m.group("key") in out:
                raise ValueError(f"duplicate addendum key {m.group('key')!r}")
            out[m.group("key")] = to_text(m.group("val"))
    if not out:
        raise ValueError(f"no {PREFIX} macro found in {NUMBERS}")
    return Macros(out)


def _need(m: Macros, key: str, *allowed: str) -> None:
    if m[key] not in allowed:
        raise ValueError(f"{key} is {m[key]!r}; the README text assumes one of {allowed}")


def _d(m: Macros, key: str) -> str:
    return f"{m[key + '.d']} pp [{m[key + '.lo']}, {m[key + '.hi']}]"


def _cs(m: Macros, key: str) -> str:
    """A contrast: difference, bootstrap interval, seed-level interval."""
    return (f"{m[key + '.b.d']} pp (bootstrap [{m[key + '.b.lo']}, {m[key + '.b.hi']}], "
            f"seed-level [{m[key + '.s.lo']}, {m[key + '.s.hi']}])")


def block(m: Macros) -> str:
    """The Markdown between (and including) the markers."""
    _need(m, "x1.prim.sep", "separated (−)")
    _need(m, "x1.prim.f.sep", "separated (−)")
    _need(m, "x1.cu.nsep", "0")
    for k in ("h2bcmlp", "h3"):
        _need(m, f"x3.{k}.verdict", ROBUST)
        _need(m, f"l1.can.{k}.s.sep", NOT_SEP)
        _need(m, f"x2.{k}.verdict", ROBUST)
    for k in ("h2bcrnn", "h2c"):
        _need(m, f"x3.{k}.verdict", NOT_SEP)
    _need(m, "x2.h2bcrnn.verdict", ROBUST)
    _need(m, "x2.h2bcrnn.relation", "new separation")
    _need(m, "x2.h2c.verdict", NOT_SEP)
    _need(m, "l1.can.h2c.b.sep", NOT_SEP)
    lines = [
        BEGIN,
        "",
        f"![Left: X1 on Lift, success on the base and arm-offset robustness sets for diffusion, "
        "BC-MLP-chunk and BC-MLP-chunk with the diffusion sampler's action bound. Right: the "
        "four declared Can contrasts for the frozen Level 1 Can PH result, the X3 replication "
        f"and X2 (Can MH), each with its bootstrap interval, side by side.]({CHART})",
        "",
        "Three follow-up studies ran after v1.0.0, each preregistered and run once. None of "
        "them changes a frozen number above. Each keeps its own label. Intervals are "
        f"{m['level']} %; \"not separated\" is never read as equality.",
        "",
        "- **X1, the action bound (exploratory, after the freeze; validation regime, the "
        "Session 7 robustness starts).** The frozen BC checkpoints were re-run with the "
        f"diffusion sampler's action bound ({m['x1.nevals']} evaluations).",
        f"  - Under the Lift arm offset, drop(diffusion) − drop(clamped BC-MLP-chunk) was "
        f"{_d(m, 'x1.prim')}, {m['x1.prim.sep']}; the frozen, unclamped value is "
        f"{m['x1.prim.f.d']} pp.",
        f"  - The clamp changed {m['x1.lift.chg.chunkc.arm']} % of BC-MLP-chunk's actions "
        f"there. Yet {m['x1.cu.nsep']} of {m['x1.cu.n']} clamped − unclamped success "
        "differences separated.",
        "  - So BC's missing bound does not explain the robustness gaps. The diffusion side of "
        "KI-052, clipping inside the sampler, was not tested.",
        f"- **X3, more seeds on Can (separate replication).** {m['x3.nseeds']} seeds per "
        f"family on {m['x3.nstates']} fresh held-out starts.",
        f"  - H2 diffusion − BC-MLP {_cs(m, 'x3.h2bcmlp')} and H3 BC-RNN − BC-MLP "
        f"{_cs(m, 'x3.h3')}: {ROBUST}.",
        "  - The frozen Can separations that were fragile with "
        f"{m['l1.can.fam.diff.nseeds']} seeds (KI-053) hold.",
        f"  - H2 diffusion − BC-RNN ({m['x3.h2bcrnn.b.d']} pp) and H2-c ({m['x3.h2c.b.d']} pp) "
        "stay not separated, as frozen.",
        "- **X2 (Can MH), mixed-quality demonstrations from several operators.**",
        "  - Reported beside the frozen Level 1 Can PH result, never pooled with it, and "
        "within-operator only.",
        f"  - X2 episodes end at {m['x2.horizon']} steps and Level 1's at {m['l1.horizon']}, so "
        "the success scales differ.",
        f"  - Success: diffusion {m['x2.fam.diff.succ']} %, BC-MLP-chunk "
        f"{m['x2.fam.chunk.succ']} %, BC-RNN {m['x2.fam.bcrnn.succ']} %, BC-MLP "
        f"{m['x2.fam.bcmlp.succ']} %.",
        f"  - H2 diffusion − BC-RNN {_cs(m, 'x2.h2bcrnn')}: {ROBUST}, a new separation (not "
        "separated on Can PH).",
        f"  - H2 diffusion − BC-MLP {m['x2.h2bcmlp.b.d']} pp and H3 {m['x2.h3.b.d']} pp: "
        f"{ROBUST}.",
        f"  - H2-c diffusion − BC-MLP-chunk {_cs(m, 'x2.h2c')}: not separated.",
        "",
        "On Can, in the frozen result, X3 and X2 alike, no primary analysis separates the "
        "diffusion policy from the deterministic MLP that sees the same history and predicts "
        "the same action chunks (H2-c). Details, tables and caveats are in the [v1.1 addendum]"
        "(paper/addendum/addendum.pdf) (a local build, not peer reviewed).",
        "",
        END,
    ]
    return "\n".join(lines)


def render(text: str, m: Macros) -> str:
    """``text`` with the extensions block rewritten; exactly one block must exist."""
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("README must contain exactly one extensions BEGIN and END marker")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return head + block(m) + tail


def check(root: Path = REPO_ROOT) -> list[str]:
    """Problems with the committed block and chart (empty when both equal a fresh build)."""
    problems = []
    text = (root / README).read_text(encoding="utf-8")
    try:
        if render(text, load(root)) != text:
            problems.append(f"{README}: the {NAME} block differs from a fresh generation "
                            "(run make readme-extensions)")
    except (ValueError, KeyError, FileNotFoundError) as exc:
        problems.append(f"{README}: {exc}")
    index = json.loads((root / OUT / "index.json").read_text())
    want = {o["path"]: o["sha256"] for o in index["outputs"]}.get(CHART)
    got = hashlib.sha256((root / CHART).read_bytes()).hexdigest() \
        if (root / CHART).exists() else None
    if want is None or got != want:
        problems.append(f"{CHART}: differs from the addendum index (run make addendum-assets)")
    return problems
