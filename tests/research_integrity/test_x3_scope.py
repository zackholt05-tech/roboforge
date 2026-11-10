"""The X3 held-out starts are fresh and kept apart (DECISIONS D-111; extensions README rule 5).

- The X3 seed block lies outside every reserved block of the frozen studies and outside the X1
  development block.
- Once generated, the suite has exactly its declared seeds and is disjoint by state hash from
  every other committed Can suite. Only open manifests (hashes, no state values) of those
  suites are read.
- The frozen code files the X3 seeds under ``eval_adhoc`` (the reason for the partition label).
- Only the X3 modules and scripts name the X3 suite.
"""

from __future__ import annotations

import pytest

from roboforge.paths import REPO_ROOT
from roboforge.replication import x3_states as x3
from roboforge.sim import eval_states as es
from roboforge.sim import robustness as rb

X1_DEVELOPMENT_SEEDS = (13_000_000, 13_000_005)  # D-107, 5 development starts


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def test_x3_block_is_outside_every_frozen_and_x1_block() -> None:
    cfg = x3.x3_config()
    block = tuple(cfg["reserved_seeds"])
    used = (cfg["seed_base"], cfg["seed_base"] + cfg["n_states"])
    assert block[0] <= used[0] < used[1] <= block[1]
    frozen = [tuple(s["reserved_seeds"]) for s in es.suites_config()["suites"].values()]
    frozen.append(tuple(rb.robustness_config()["base"]["reserved_seeds"]))
    assert len(frozen) == 3
    for other in (*frozen, X1_DEVELOPMENT_SEEDS):
        assert not _overlap(block, other), f"X3 block {block} overlaps {other}"


def test_frozen_code_files_the_x3_seeds_as_ad_hoc() -> None:
    cfg = x3.x3_config()
    for seed in (cfg["seed_base"], cfg["seed_base"] + cfg["n_states"] - 1):
        assert es.suite_of_seed(seed) is None
        assert es.check_seed_allowed(cfg["task"], seed) == "eval_adhoc" == cfg["partition"]


def test_x3_suite_is_fresh_and_disjoint() -> None:
    cfg = x3.x3_config()
    task, name = cfg["task"], cfg["suite"]
    if not es.manifest_path(task, name).exists():
        pytest.skip("X3 starts not generated yet")
    m = es.read_manifest(task, name)
    assert (m["suite"], m["partition"], m["n_states"]) == (name, "eval_adhoc", 100)
    assert (m["decision"], m["label"]) == ("D-111", "separate replication")
    assert m["config_sha256"] == x3.config_sha256()
    assert [s["seed"] for s in m["states"]] == list(range(cfg["seed_base"],
                                                          cfg["seed_base"] + 100))
    mine = {s["state_sha256"] for s in m["states"]}
    assert len(mine) == 100
    for suite in ("validation", es.LOCKED_SUITE, *rb.sets_for(task)):
        other = {s["state_sha256"] for s in es.read_manifest(task, suite)["states"]}
        assert not mine & other, f"X3 shares a start with {task}/{suite}"
    assert es.verify_data_file(task, name)


def test_only_x3_code_names_the_x3_suite() -> None:
    name = x3.suite_name()
    allowed = ("src/roboforge/replication/", "scripts/generate_x3_states.py",
               "scripts/eval_x3.py", "scripts/run_x3.py", "scripts/analyze_x3.py",
               "scripts/check_x3_engineering.py")
    offenders = []
    for root in ("src", "scripts"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if name in path.read_text() and not rel.startswith(allowed):
                offenders.append(rel)
    assert offenders == [], f"files naming the X3 suite: {offenders}"
