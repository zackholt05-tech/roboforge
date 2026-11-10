"""The paper's LaTeX, bibliography and post-hoc analyses (DECISIONS D-094 to D-096).

- the sections contain no typed numbers or placeholders and use only defined macros;
- every bibliography entry has a DOI, arXiv id or URL and passed the recorded resolution check;
- a hash-chained summary that differs from the hash recorded in its pinned parent stops the
  build;
- the seed-level t intervals (D-095) match textbook formulas, and the labelled tables keep the
  frozen verdict beside them;
- the secondary metrics (D-096) are plain means of the committed per-episode values.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import pytest
from scipy import stats

from roboforge.paper import lint, secondary, sensitivity
from roboforge.paper.sources import SOURCES, SourceError, Sources
from roboforge.paths import REPO_ROOT

BIB = REPO_ROOT / "paper/references.bib"
BIBCHECK = REPO_ROOT / "paper/bibliography_check.json"


def test_paper_lint_is_clean() -> None:
    assert lint.lint_paper() == []


def test_lint_catches_typed_numbers_and_placeholders() -> None:
    assert lint.bare_numbers("success was 97.3 percent") == ["97.3"]
    assert lint.bare_numbers("H2-c and KI-049 and Level~1 and \\rf{l1.git}") == []
    assert lint.bare_numbers("$K = 100$")
    assert lint.PLACEHOLDER.search("result TBD")


def test_every_used_macro_is_defined() -> None:
    defined = lint.defined_keys()
    for path, keys in lint.used_keys().items():
        assert keys <= defined, f"{path}: undefined {sorted(keys - defined)}"


def _bib_entries() -> dict[str, str]:
    txt = BIB.read_text()
    return {m.group(1): m.group(2) for m in
            re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", txt, re.S)}


def test_every_bib_entry_is_identified_and_was_resolved() -> None:
    entries = _bib_entries()
    check = {e["key"]: e for e in json.loads(BIBCHECK.read_text())["entries"]}
    for key, body in entries.items():
        assert re.search(r"\b(doi|eprint|url)\s*=", body), f"{key}: no DOI, arXiv id or URL"
        assert key in check and check[key]["ok"], f"{key}: not resolved in {BIBCHECK.name}"
    assert set(check) == set(entries)


def test_every_citation_has_an_entry() -> None:
    cited: set[str] = set()
    for p in (REPO_ROOT / "paper/sections").glob("*.tex"):
        for m in re.finditer(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", p.read_text()):
            cited |= {k.strip() for k in m.group(1).split(",")}
    assert cited <= set(_bib_entries())


def test_an_edited_chained_summary_stops_the_build(tmp_path: Path) -> None:
    parent = SOURCES["l1_results"]
    pc = json.loads((REPO_ROOT / parent.path).read_text())["tasks"]["lift"]["per_checkpoint"][0]
    for rel in (parent.path, pc["summary"]):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    src = Sources(root=tmp_path)
    assert src.chained_json("l1_results", pc["summary"], pc["summary_sha256"])["episodes"]
    p = tmp_path / pc["summary"]
    p.write_bytes(p.read_bytes().replace(b'"success": true', b'"success": false', 1))
    with pytest.raises(SourceError):
        Sources(root=tmp_path).chained_json("l1_results", pc["summary"], pc["summary_sha256"])


def test_welch_and_paired_match_textbook_formulas() -> None:
    a, b = [0.68, 0.88, 0.74], [0.92, 0.98, 0.96]
    (lo, hi), df = sensitivity.welch(b, a)
    va, vb = (sum((x - sum(v) / 3) ** 2 for x in v) / 2 for v in (a, b))
    se = math.sqrt(va / 3 + vb / 3)
    df_ref = (va / 3 + vb / 3) ** 2 / ((va / 3) ** 2 / 2 + (vb / 3) ** 2 / 2)
    h = stats.t.ppf(0.975, df_ref) * se
    d = sum(b) / 3 - sum(a) / 3
    assert df == pytest.approx(df_ref)
    assert (lo, hi) == pytest.approx((d - h, d + h))
    (plo, phi), pdf = sensitivity.paired([1.0, 0.98, 1.0], [0.98, 0.98, 1.0])
    assert pdf == 2 and plo < 0 < phi
    assert sensitivity.welch([1.0, 1.0, 1.0], [0.0]) == (None, None)


def test_sensitivity_keeps_every_frozen_verdict() -> None:
    rows = sensitivity.rows(Sources())
    r = json.loads((REPO_ROOT / SOURCES["l1_results"].path).read_text())
    frozen = {(t, d["a"], d["b"]): d for t in ("lift", "can")
              for d in r["tasks"][t]["success"]["differences"]}
    l1 = [x for x in rows if x.analysis.startswith("Level 1")]
    assert len(l1) == len(frozen)
    for x in l1:
        t = x.analysis.split(", ")[1].lower()
        match = [d for (tt, _, _), d in frozen.items() if tt == t
                 and d["difference"] == x.frozen and tuple(d["ci95"]) == x.frozen_ci]
        assert match and match[0]["separation"] == x.frozen_sep
    table = sensitivity.table(rows)
    assert table.count("\\\\") >= len(rows)


def test_secondary_metrics_are_plain_means_of_the_summaries() -> None:
    src = Sources()
    data = secondary.rows(src)
    r = json.loads((REPO_ROOT / SOURCES["l1_results"].path).read_text())
    for t in ("lift", "can"):
        pcs = [pc for pc in r["tasks"][t]["per_checkpoint"] if pc["family"] == "bc_mlp"]
        eps = [e for pc in pcs
               for e in json.loads((REPO_ROOT / pc["summary"]).read_text())["episodes"]]
        mean = sum(e["gripper_sign_changes"] for e in eps) / len(eps)
        assert data[t]["bc_mlp"]["sign_mean"] == pytest.approx(mean)
        assert data[t]["bc_mlp"]["n"] == len(eps) == 150
