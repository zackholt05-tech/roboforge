"""The final-test initial states are saved and hashed but locked (MASTER_SPEC §3.2, Session 2).

These tests need no simulator: they read only the committed open manifests and the code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboforge.paths import REPO_ROOT
from roboforge.sim import eval_states as es

TASKS = ("lift", "can")
# Only the suite module, the snapshot partition labels and the generator may name the suite.
ALLOWED = {"src/roboforge/sim/eval_states.py", "src/roboforge/sim/snapshot.py",
           "scripts/generate_eval_states.py"}


def test_freeze_file_does_not_exist_yet() -> None:
    assert not (REPO_ROOT / "docs" / "FINAL_TEST_FREEZE.md").exists()


@pytest.mark.parametrize("task", TASKS)
def test_loading_final_test_states_is_refused(task: str) -> None:
    unlocked, why = es.lock_status(task)
    assert not unlocked and "FINAL_TEST_FREEZE" in why
    with pytest.raises(es.FinalTestLockedError):
        es.load_suite(task, es.LOCKED_SUITE)


@pytest.mark.parametrize("seed", [900000, 900049, 950000, 999999])
def test_resetting_to_a_final_test_seed_is_refused(seed: int) -> None:
    with pytest.raises(es.FinalTestLockedError):
        es.check_seed_allowed("lift", seed)


@pytest.mark.parametrize(("seed", "partition"), [(100000, "eval_validation"),
                                                 (199999, "eval_validation"),
                                                 (0, "eval_adhoc"), (1_000_000, "eval_adhoc")])
def test_other_seeds_map_to_their_partition(seed: int, partition: str) -> None:
    assert es.check_seed_allowed("lift", seed) == partition


@pytest.mark.parametrize("task", TASKS)
def test_suites_are_saved_hashed_and_disjoint(task: str) -> None:
    val, test = es.read_manifest(task, "validation"), es.read_manifest(task, "final_test")
    assert val["n_states"] == test["n_states"] == 50
    assert val["partition"] == "eval_validation" and test["partition"] == "eval_final_test"
    assert test["locked_until"] == "docs/FINAL_TEST_FREEZE.md"
    seeds_v = {s["seed"] for s in val["states"]}
    seeds_t = {s["seed"] for s in test["states"]}
    assert seeds_v == set(range(100000, 100050)) and seeds_t == set(range(900000, 900050))
    assert not ({s["state_sha256"] for s in val["states"]}
                & {s["state_sha256"] for s in test["states"]})
    for m in (val, test):  # open manifests carry hashes, never state values
        assert all(set(s) == {"state_id", "index", "seed", "model_xml_sha256", "state_sha256"}
                   for s in m["states"])
        assert es.verify_data_file(task, m["suite"])


@pytest.mark.parametrize("task", TASKS)
def test_partition_index_identifies_train_validation_test(task: str) -> None:
    committed = json.loads((es.suite_dir(task) / "partitions.json").read_text())
    assert committed == es.partitions_index(task)
    assert committed["demonstrations"]["demo_train"]["n"] == 180
    assert committed["demonstrations"]["demo_valid"]["n"] == 20
    assert committed["eval_suites"]["final_test"]["locked"] is True
    assert committed["eval_suites"]["validation"]["locked"] is False


def test_no_other_code_names_the_locked_suite() -> None:
    offenders = []
    for root in ("src", "scripts"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel not in ALLOWED and "final_test" in path.read_text():
                offenders.append(rel)
    assert offenders == [], f"files referencing the locked suite: {offenders}"


def test_locked_data_file_is_only_read_inside_the_suite_module() -> None:
    text = Path(es.__file__).read_text()
    assert text.count("lzma.decompress(") == 3  # verify_data_file, load_suite, _max_state_diff
