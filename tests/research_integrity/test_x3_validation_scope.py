"""The X3 validation driver only evaluates and selects (DECISIONS D-112).

It never trains or resumes a run, never reuses the frozen suite orchestrator (which would
treat the X3 runs as pending training), never names a held-out suite, and launches only the
frozen validation evaluator and selection script.
"""

from __future__ import annotations

from roboforge.paths import REPO_ROOT

FILES = ("scripts/x3_validate.py", "src/roboforge/replication/x3_validation.py")


def test_the_driver_never_trains_or_reuses_the_suite() -> None:
    for rel in FILES:
        src = (REPO_ROOT / rel).read_text()
        for word in ("run_suite import", "import run_suite", "Suite(", "launch_train",
                     "--resume", "heldout", "x3_states"):
            assert word not in src, (rel, word)
    src = (REPO_ROOT / FILES[1]).read_text()
    assert src.count('"scripts/eval_policy.py"') == 1
    assert src.count('"scripts/select_checkpoint.py"') == 1
    # train argv is used only to recognise a live training process
    drv = (REPO_ROOT / FILES[0]).read_text()
    assert "train_argv" not in drv and "scripts/train.py" not in drv
