"""The Session 7 robustness starts (RESEARCH_PROTOCOL §14.1-14.2, D-063) are disjoint from the
validation and held-out starts, by state hash and by seed block. Reads only open manifests."""

from __future__ import annotations

import pytest

from roboforge.sim import eval_states as es
from roboforge.sim import robustness as rb

TASKS = ("lift", "can")


@pytest.mark.parametrize("task", TASKS)
def test_robustness_starts_are_disjoint_from_validation_and_test(task: str) -> None:
    if not es.manifest_path(task, rb.base_name()).exists():
        pytest.skip("robustness starts not generated yet")
    others = {s["state_sha256"] for suite in ("validation", "final_test")
              for s in es.read_manifest(task, suite)["states"]}
    seen: set[str] = set()
    for name in rb.sets_for(task):
        m = es.read_manifest(task, name)
        hashes = {s["state_sha256"] for s in m["states"]}
        assert m["partition"] == "eval_robustness" and len(hashes) == 50
        assert not hashes & others, f"{task}/{name} shares a start with validation or test"
        assert not hashes & seen, f"{task}/{name} repeats a start of another robustness set"
        seen |= hashes
        assert {s["seed"] for s in m["states"]} == set(range(300000, 300050))
