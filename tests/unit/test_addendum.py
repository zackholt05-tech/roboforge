"""The v1.1 addendum, README block and chart come only from the three pinned results (D-133).

- Each source is refused unless its bytes equal the pin.
- Generated values equal the studies' own ``tables.md`` digit for digit (spot checks of every
  study's headline row).
- Generated tables hold no literal number: every value is an ``\\ax{}`` macro.
- The committed generated files, the README block and the chart equal a fresh build, and the
  addendum prose passes its lint.
- The README generator refuses to write prose whose verdict no longer holds.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from roboforge.addendum import build, numbers, readme, sources, tables
from roboforge.paths import REPO_ROOT
from roboforge.release.macros import Macros, to_text


@pytest.fixture(scope="module")
def macros() -> dict[str, str]:
    n = numbers.build_numbers(sources.load_all())
    return {k: to_text(v).replace("−", "-") for k, v in n.values.items()}


def test_every_source_equals_its_pin() -> None:
    assert sources.verify() == []


def test_a_changed_source_is_refused(tmp_path: Path) -> None:
    for s in sources.SOURCES.values():
        dst = tmp_path / s.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / s.path, dst)
    p = tmp_path / sources.SOURCES["x2"].path
    p.write_bytes(p.read_bytes() + b" ")
    x2 = sources.SOURCES["x2"]
    assert sources.verify(tmp_path) == [f"{x2.path}: SHA-256 differs from the pin {x2.sha256}"]
    with pytest.raises(RuntimeError):
        sources.load_all(tmp_path)


def _table(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_values_equal_the_study_tables(macros: dict[str, str]) -> None:
    x1 = _table("reports/session13/x1_clamp/tables.md")
    m = macros
    assert (f"**{m['x1.prim.d']} pp [{m['x1.prim.lo']}, {m['x1.prim.hi']}], "
            f"{m['x1.prim.sep']}**") in x1
    assert f"{m['x1.lift.chunkc.arm.succ']} [{m['x1.lift.chunkc.arm.lo']}, " \
           f"{m['x1.lift.chunkc.arm.hi']}]" in x1
    x3 = _table("reports/session14/x3/tables.md")
    assert (f"| H2 diffusion − BC-MLP | {m['l1.can.h2bcmlp.b.d']} pp "
            f"[{m['l1.can.h2bcmlp.b.lo']}, {m['l1.can.h2bcmlp.b.hi']}], "
            f"{m['l1.can.h2bcmlp.b.sep']} | [{m['l1.can.h2bcmlp.s.lo']}, "
            f"{m['l1.can.h2bcmlp.s.hi']}] | {m['x3.h2bcmlp.b.d']} pp |") in x3
    x2 = _table("reports/session17/x2/tables.md")
    assert (f"| H2 diffusion − BC-RNN | {m['l1.x2side.h2bcrnn.b.d']} pp") in x2
    assert (f"| {m['x2.h2bcrnn.b.d']} pp | [{m['x2.h2bcrnn.b.lo']}, {m['x2.h2bcrnn.b.hi']}] | "
            f"[{m['x2.h2bcrnn.s.lo']}, {m['x2.h2bcrnn.s.hi']}] (") in x2
    assert f"| diffusion | {m['x2.fam.diff.succ']} | [{m['x2.fam.diff.lo']}, " \
           f"{m['x2.fam.diff.hi']}] | {m['x2.fam.diff.perseed']} |" in x2
    assert m["x2.horizon"] == "500" and m["l1.horizon"] == "400"


def test_tables_hold_no_literal_number() -> None:
    for name, fn in tables.TABLES.items():
        t = fn()
        t = re.sub(r"\\(?:ax|file|code)\{[^}]*\}", " ", t)
        t = re.sub(r"\{(?:p\{[\d.]+cm\}|l)+\}|\[\d+pt\]|p\{[\d.]+cm\}|Level~1", " ", t)
        t = re.sub(r"\b(?:X[123]|H[1-5]|SHA-256)\b", " ", t)  # identifiers
        assert not re.search(r"\d", t), f"{name}: literal number in {t!r}"


def test_committed_outputs_equal_a_fresh_build_and_lint_passes() -> None:
    assert build.check(build.build_all()) == []
    assert build.lint() == []


def test_readme_block_and_chart_equal_a_fresh_build() -> None:
    assert readme.check() == []


def test_readme_refuses_prose_whose_verdict_changed() -> None:
    m = readme.load()
    values = {k: m[k] for k in m._v}  # noqa: SLF001 - a copy of the strict accessor's values
    values["x2.h2c.verdict"] = "robust (+)"
    with pytest.raises(ValueError, match="x2.h2c.verdict"):
        readme.block(Macros(values))
