"""X1 never reaches the held-out or validation states (DECISIONS D-107; extensions README rule 5).

The X1 runner admits only the Session 7 robustness sets of a task. It refuses the locked
held-out suite and the validation suite before it creates any directory or loads any
checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from roboforge.exploratory import clamp as xc
from roboforge.paths import REPO_ROOT
from roboforge.sim import eval_states as es
from roboforge.sim import robustness as rb

TASKS = ("lift", "can")


@pytest.mark.parametrize("task", TASKS)
def test_only_robustness_sets_are_admitted(task: str) -> None:
    for suite in rb.sets_for(task):
        xc.check_x1_suite(task, suite)
    for suite in (es.LOCKED_SUITE, "validation", "anything_else"):
        with pytest.raises(ValueError):
            xc.check_x1_suite(task, suite)


@pytest.mark.parametrize("task", TASKS)
def test_the_runner_refuses_the_held_out_suite_before_touching_anything(
        task: str, tmp_path: Path) -> None:
    run = tmp_path / "train"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "t", "status": "complete", "config": {"task": task, "family": "bc_mlp"},
        "checkpoints": [{"step": 1, "sha256": "a" * 64, "path": str(run / "missing.pt")}]}))
    runs_dir = tmp_path / "runs"
    with pytest.raises(ValueError):
        xc.evaluate_x1(run, 1, es.LOCKED_SUITE, family="bc_mlp", expected_sha256="a" * 64,
                       clamp=True, runs_dir=runs_dir)
    assert not runs_dir.exists()


def test_the_orchestrator_plans_robustness_sets_only() -> None:
    spec = importlib.util.spec_from_file_location(
        "run_x1_clamp", REPO_ROOT / "scripts" / "run_x1_clamp.py")
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)
    for j in rx.jobs():
        assert j["suite"] in rb.sets_for(j["task"])
        assert j["suite"] != es.LOCKED_SUITE
