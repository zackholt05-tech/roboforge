"""The X2 suites are fresh and kept apart; the X2 held-out suite is named only by its generator
(DECISIONS D-123; extensions README rule 5; X2_harder_task.md §3).

- Both X2 seed blocks lie outside every reserved block of the frozen studies, the X1
  development block and the X3 block, and outside each other.
- The frozen code files the X2 seeds under ``eval_adhoc`` (the reason for the partition label).
- Once generated, each suite has exactly its declared seeds and is disjoint by state hash from
  every other committed Can suite and from the other X2 suite. Only open manifests (hashes, no
  state values) are read.
- Only the X2 suite module and its generator name the X2 held-out suite, so no X2-a code path
  can roll out on it.
"""

from __future__ import annotations

import pytest

from roboforge.paths import REPO_ROOT
from roboforge.replication import x3_states
from roboforge.sim import eval_states as es
from roboforge.sim import robustness as rb
from roboforge.x2 import states as x2s

X1_DEVELOPMENT_SEEDS = (13_000_000, 13_000_005)  # D-107, 5 development starts
# Files allowed to name the X2 held-out suite in X2-a (D-123).
HELDOUT_NAMERS = ("src/roboforge/x2/states.py", "scripts/generate_x2_states.py")


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _blocks() -> dict[str, tuple[int, int]]:
    return {n: tuple(s["reserved_seeds"]) for n, s in x2s.x2_suites_config()["suites"].items()}


def test_x2_blocks_are_outside_every_other_block() -> None:
    frozen = [tuple(s["reserved_seeds"]) for s in es.suites_config()["suites"].values()]
    frozen.append(tuple(rb.robustness_config()["base"]["reserved_seeds"]))
    assert len(frozen) == 3
    others = [*frozen, X1_DEVELOPMENT_SEEDS, tuple(x3_states.x3_config()["reserved_seeds"])]
    blocks = _blocks()
    for name, block in blocks.items():
        for other in others:
            assert not _overlap(block, other), f"X2 {name} block {block} overlaps {other}"
    a, b = blocks.values()
    assert not _overlap(a, b)


def test_frozen_code_files_the_x2_seeds_as_ad_hoc() -> None:
    cfg = x2s.x2_suites_config()
    for s in cfg["suites"].values():
        for seed in (s["seed_base"], s["seed_base"] + s["n_states"] - 1):
            assert es.suite_of_seed(seed) is None
            assert es.check_seed_allowed(cfg["task"], seed) == "eval_adhoc" == cfg["partition"]


@pytest.mark.parametrize("name", [x2s.VALIDATION_SUITE, x2s.HELDOUT_SUITE])
def test_x2_suite_is_fresh_and_disjoint(name: str) -> None:
    cfg = x2s.x2_suites_config()
    task, s = cfg["task"], cfg["suites"][name]
    if not es.manifest_path(task, name).exists():
        pytest.skip("X2 suites not generated yet")
    m = es.read_manifest(task, name)
    assert (m["suite"], m["partition"], m["n_states"]) == (name, "eval_adhoc", s["n_states"])
    assert (m["decision"], m["config_sha256"]) == ("D-123", x2s.config_sha256())
    assert [st["seed"] for st in m["states"]] == list(range(s["seed_base"],
                                                            s["seed_base"] + s["n_states"]))
    mine = {st["state_sha256"] for st in m["states"]}
    assert len(mine) == s["n_states"]
    others = ["validation", es.LOCKED_SUITE, *rb.sets_for(task), x3_states.suite_name(),
              *(o for o in cfg["suites"] if o != name)]
    for suite in others:
        if suite in cfg["suites"] and not es.manifest_path(task, suite).exists():
            continue
        other = {st["state_sha256"] for st in es.read_manifest(task, suite)["states"]}
        assert not mine & other, f"X2 {name} shares a start with {task}/{suite}"
    assert es.verify_data_file(task, name)


def test_only_the_x2_suite_code_names_the_x2_heldout_suite() -> None:
    offenders = []
    for root in ("src", "scripts"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if x2s.HELDOUT_SUITE in path.read_text() and rel not in HELDOUT_NAMERS:
                offenders.append(rel)
    assert offenders == [], f"files naming the X2 held-out suite: {offenders}"
