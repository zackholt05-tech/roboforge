"""X2-b stays on X2 validation starts and keeps X2-a's evidence untouched (D-125 to D-127).

- Every X2-b evaluation goes through the X2-a evaluator with the configured suite, which is the
  X2 validation suite; the evaluator refuses any other suite (``check_x2_eval_suite``).
- No X2-b source names the X2 held-out suite (the X2-a scope test also scans every source).
- X2-b writes summaries only under ``reports/session16``; it reads the X2-a pilot summaries.
- The X2-a pilot runs are never retrained, and X2-b's run prefix is new.
- The X2-b search space is exactly Level 1's Can trials (config hashes equal), so the
  per-family tuning budget is the frozen equal one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roboforge.paths import REPO_ROOT
from roboforge.train.suite import load_plan
from roboforge.x2 import states as x2s
from roboforge.x2 import suite as xs
from roboforge.x2.evaluate import check_x2_eval_suite

X2B_SOURCES = ("src/roboforge/x2/suite.py", "src/roboforge/x2/host.py", "scripts/x2b_train.py",
               "scripts/x2b_eval.py", "scripts/x2b_suite.py", "scripts/x2b_report.py",
               "scripts/x2b_benchmark.py")
FROZEN_PREFIXES = ("s6", "s7", "s14", "s15x2")


def test_x2b_sources_exist_and_never_name_the_heldout_suite() -> None:
    for rel in X2B_SOURCES:
        text = (REPO_ROOT / rel).read_text()
        assert x2s.HELDOUT_SUITE not in text, rel
        assert "HELDOUT_SUITE" not in text, rel


def test_x2b_evaluates_only_the_validation_suite() -> None:
    cfg = xs.x2b_config()
    assert cfg["suite"] == x2s.VALIDATION_SUITE
    check_x2_eval_suite("can", cfg["suite"])
    with pytest.raises(ValueError):
        check_x2_eval_suite("can", x2s.HELDOUT_SUITE)
    text = (REPO_ROOT / "scripts/x2b_eval.py").read_text()
    assert 'suite=x2b_config()["suite"]' in text


def test_x2b_writes_only_session16_summaries() -> None:
    cfg = xs.x2b_config()
    assert cfg["summary_dirs"][-1].startswith("reports/session16/")
    assert cfg["out_dir"].startswith("reports/session16/")
    assert cfg["summary_dirs"][0] == "reports/session15/x2/pilot/evals"


def test_pilot_runs_are_not_retrained_and_the_prefix_is_new() -> None:
    plan = xs.x2b_plan()
    assert plan.prefix not in FROZEN_PREFIXES
    text = (REPO_ROOT / "scripts/x2b_train.py").read_text()
    assert "if run.pilot:" in text and "return 2" in text


def test_search_space_equals_level1_can() -> None:
    plan = xs.x2b_plan()
    level1 = load_plan(Path(REPO_ROOT / "configs/experiments/level1_core_can.yaml"))
    for fam in plan.families:
        other = next(f for f in level1.families if f.name == fam.name)
        assert fam.trials == other.trials and fam.config_path == other.config_path
    assert plan.budget == level1.budget and plan.tuning_seed == level1.tuning_seed == 0
    assert plan.replicate_seeds == level1.replicate_seeds
