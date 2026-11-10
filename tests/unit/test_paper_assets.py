"""Paper assets come from the pinned frozen files and reproduce their values (DECISIONS D-094).

- every source exists and has its pinned SHA-256; an edited source stops the build;
- a fresh in-memory build equals the committed ``paper/generated`` byte for byte;
- every generated value equals the frozen ``tables.md`` value at the same rounding and sign;
- the index cites every source with its hash, and never the Observatory copy or ``runs/``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from roboforge.paper import build, fmt
from roboforge.paper.numbers import L1_FAMILIES, TASKS, build_numbers
from roboforge.paper.sources import SOURCES, SourceError, Sources
from roboforge.paths import REPO_ROOT


@pytest.fixture(scope="module")
def files() -> dict[str, bytes]:
    return build.build_all()


@pytest.fixture(scope="module")
def nums() -> dict[str, str]:
    return build_numbers(Sources()).values


def _plain(v: str) -> str:
    """Undo the LaTeX sign typesetting: back to the frozen tables' ASCII form."""
    return v.replace("\\ensuremath{-}", "-").replace("\\ensuremath{+}", "+")


def test_every_source_matches_its_pin() -> None:
    assert Sources().verify_all() == []
    assert not any("findings.json" in s.path or s.path.startswith("runs/")
                   for s in SOURCES.values())


def test_an_edited_source_stops_the_build(tmp_path: Path) -> None:
    s = SOURCES["l1_results"]
    dst = tmp_path / s.path
    dst.parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / s.path, dst)
    dst.write_bytes(dst.read_bytes().replace(b'"estimate": 0.9', b'"estimate": 0.8', 1))
    with pytest.raises(SourceError):
        Sources(root=tmp_path).json("l1_results")


def test_committed_outputs_equal_a_fresh_build(files: dict[str, bytes]) -> None:
    assert build.check(files) == []


def test_index_cites_every_output_and_source(files: dict[str, bytes]) -> None:
    index = json.loads(files[f"{build.OUT}/index.json"])
    assert index["schema"] == build.INDEX_SCHEMA
    listed = {e["path"] for e in index["outputs"]}
    assert listed == set(files) - {f"{build.OUT}/index.json"}
    pins = {s.path: s.sha256 for s in SOURCES.values()}
    for e in index["outputs"]:
        assert e["sources"], e["path"]
        for s in e["sources"]:
            if e["kind"] != "render":
                assert pins[s["path"]] == s["sha256"]
        assert e["sha256"] == __import__("hashlib").sha256(files[e["path"]]).hexdigest()


def test_tables_and_numbers_carry_no_timestamp(files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        if rel.endswith(".tex"):
            assert not re.search(rb"20\d\d-\d\d-\d\dT", data), rel


def test_level1_values_equal_the_frozen_tables(nums: dict[str, str]) -> None:
    txt = (REPO_ROOT / SOURCES["l1_tables"].path).read_text()
    r = json.loads((REPO_ROOT / SOURCES["l1_results"].path).read_text())
    sections = dict(zip(TASKS, txt.split("## can"), strict=True))
    for t in TASKS:
        sec = sections[t]
        for f in L1_FAMILIES:
            k = f"l1.{t}.{fmt.FAMILY_KEY[f]}"
            seeds = ", ".join(nums[f"{k}.s{i}"] for i in range(len(
                r["tasks"][t]["success"]["families"][f]["per_seed"])))
            row = (f"| {f} | {nums[k + '.succ']} | [{nums[k + '.lo']}, {nums[k + '.hi']}] | "
                   f"{seeds} |")
            assert row in sec, row
            lrow = f"| {f} | {nums[k + '.len']} | [{nums[k + '.len.lo']}, {nums[k + '.len.hi']}] |"
            assert lrow in sec, lrow
        for d in r["tasks"][t]["success"]["differences"]:
            k = f"l1.{t}.{fmt.FAMILY_KEY[d['a']]}-{fmt.FAMILY_KEY[d['b']]}"
            lo, hi = (f"{100 * x:.1f}" for x in d["ci95"])
            row = (f"| {d['hypothesis']} | {d['a']} − {d['b']} | {_plain(nums[k + '.d'])} | "
                   f"[{lo}, {hi}] | {_plain(nums[k + '.sep'])} |")
            assert row in sec, row
        for d in r["tasks"][t]["censored_episode_length"]["differences"]:
            k = f"l1.{t}.len.{fmt.FAMILY_KEY[d['a']]}-{fmt.FAMILY_KEY[d['b']]}"
            lo, hi = (f"{x:.1f}" for x in d["ci95"])
            row = (f"| {d['hypothesis']} | {d['a']} − {d['b']} | {_plain(nums[k + '.d'])} | "
                   f"[{lo}, {hi}] | {_plain(nums[k + '.sep'])} |")
            assert row in sec, row
        for f in ("bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet"):
            assert f"| {nums[f'l1.{t}.{fmt.FAMILY_KEY[f]}.lat']} | $0 |" in sec


def test_robustness_values_equal_the_frozen_tables(nums: dict[str, str]) -> None:
    txt = (REPO_ROOT / SOURCES["l1_robustness_tables"].path).read_text()
    a = json.loads((REPO_ROOT / SOURCES["l1_robustness"].path).read_text())
    for t in TASKS:
        b = a["tasks"][t]["h4"]["bootstrap"]
        for key_fs in b["families"]:
            f, st = key_fs.split(":")
            k = f"rob.{t}.{fmt.FAMILY_KEY[f]}.{fmt.SET_KEY[st]}"
            seeds = ", ".join(nums[f"{k}.s{i}"] for i in range(3))
            cell = f"{nums[k + '.succ']} [{nums[k + '.lo']}, {nums[k + '.hi']}] ({seeds})"
            assert cell in txt, cell
        for d in b["differences"]:
            cell = (f"| {d['a']} − {d['b']} | {d['kind']} | {100 * d['difference']:+.1f} | "
                    f"[{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] | {d['separation']} |")
            assert cell in txt, cell
        for ab in ("A1", "A2", "A3"):
            if ab in a["tasks"][t]:
                d = a["tasks"][t][ab]["bootstrap"]["differences"][0]
                assert _plain(nums[f"abl.{t}.{ab}.d"]) == f"{100 * d['difference']:+.1f}"
                line = (f"{d['a']} − {d['b']}: {_plain(nums[f'abl.{t}.{ab}.d'])} pp, 95 % CI "
                        f"[{_plain(nums[f'abl.{t}.{ab}.lo'])}, {_plain(nums[f'abl.{t}.{ab}.hi'])}]")
                assert line in txt, line


def test_rgb_values_equal_the_frozen_tables(nums: dict[str, str]) -> None:
    txt = (REPO_ROOT / SOURCES["rgb_tables"].path).read_text()
    for f in ("rgb_diffusion", "rgb_bc_chunk"):
        k = f"rgb.{fmt.FAMILY_KEY[f]}"
        assert f"| {f} | {nums[k + '.succ']} % | {nums[k + '.lo']}–{nums[k + '.hi']} % |" in txt
    line = (f"{_plain(nums['rgb.diff-bc.d'])} points, 95 % interval "
            f"[{_plain(nums['rgb.diff-bc.lo'])}"
            f", {_plain(nums['rgb.diff-bc.hi'])}]: **{nums['rgb.diff-bc.sep']}**")
    assert line in txt, line


def test_level2_values_equal_the_frozen_tables(nums: dict[str, str]) -> None:
    txt = (REPO_ROOT / SOURCES["l2_tables"].path).read_text()
    for arm in ("pc_diffusion", "rgb_diffusion"):
        for st in ("robustness_base", "ood_cube_shift"):
            k = f"l2.{fmt.FAMILY_KEY[arm]}.{fmt.SET_KEY[st]}"
            row = f"| {arm}:{st} | {nums[k + '.succ']} | {nums[k + '.lo']}–{nums[k + '.hi']} |"
            assert row in txt, row
    h5 = (f"| {_plain(nums['l2.h5.d'])} | [{_plain(nums['l2.h5.lo'])}, "
          f"{_plain(nums['l2.h5.hi'])}] | **{_plain(nums['l2.h5.sep'])}** |")
    assert h5 in txt, h5
    assert f"| {nums['l2.train.s0.msstep']} |" in txt  # the generated (uncorrected) row


def test_ki049_correction_matches_the_erratum(nums: dict[str, str]) -> None:
    errata = (REPO_ROOT / SOURCES["l2_errata"].path).read_text()
    assert f"**{nums['l2.train.s0.msstep.fixed']} ms/step**" in errata
    assert nums["l2.train.s0.msstep"] == "24.62"


def test_spend_equals_the_ledger_billing_events(nums: dict[str, str]) -> None:
    ev = [json.loads(x) for x in (REPO_ROOT / SOURCES["cost_ledger"].path).read_text().splitlines()]
    total = sum(e["billed_usd"] for e in ev if e["event"] == "billing")
    assert nums["spend.total"] == f"{total:.2f}"
    assert int(nums["spend.launches"]) == sum(e["event"] == "billing" for e in ev)
