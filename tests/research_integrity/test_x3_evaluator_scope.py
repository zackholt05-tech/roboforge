"""The X3 evaluator never reaches the locked held-out states (DECISIONS D-111).

It admits only the X3 suite of Can, or, in its engineering-check mode, only the validation
suite. It refuses anything else before it creates any directory or loads any checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboforge.replication import x3, x3_states
from roboforge.sim import eval_states as es


def test_only_the_x3_suite_is_admitted() -> None:
    task, name = x3_states.suite_task(), x3_states.suite_name()
    x3.check_x3_suite(task, name)
    x3.check_x3_suite(task, "validation", check=True)
    for suite in (es.LOCKED_SUITE, "validation", "robustness_base", "ood_arm_offset"):
        with pytest.raises(ValueError):
            x3.check_x3_suite(task, suite)
    for suite in (es.LOCKED_SUITE, name, "robustness_base"):
        with pytest.raises(ValueError):
            x3.check_x3_suite(task, suite, check=True)
    with pytest.raises(ValueError):
        x3.check_x3_suite("lift", name)


@pytest.mark.parametrize("check", [False, True])
def test_the_evaluator_refuses_the_held_out_suite_before_touching_anything(
        check: bool, tmp_path: Path) -> None:
    run = tmp_path / "train"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "t", "status": "complete", "config": {"task": "can", "family": "bc_mlp",
                                                         "seed": 0},
        "checkpoints": [{"step": 1, "sha256": "a" * 64, "path": str(run / "missing.pt")}]}))
    runs_dir = tmp_path / "runs"
    with pytest.raises(ValueError):
        x3.evaluate_x3(run, 1, es.LOCKED_SUITE, family="bc_mlp", seed=0,
                       expected_sha256="a" * 64, runs_dir=runs_dir, check=check)
    assert not runs_dir.exists()


def test_records_never_reuse_a_directory(tmp_path: Path) -> None:
    out = x3.prepare_out_dir(tmp_path, "e", check=False)
    assert out == tmp_path / "x3" / "e"
    with pytest.raises(FileExistsError):
        x3.prepare_out_dir(tmp_path, "e", check=False)
    (tmp_path / "f").mkdir()
    with pytest.raises(FileExistsError):
        x3.prepare_out_dir(tmp_path, "f", check=True)
