"""``scripts/compare_platform_evidence.py`` on small synthetic evidence directories."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from roboforge.paths import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "compare_platform_evidence", REPO_ROOT / "scripts" / "compare_platform_evidence.py")
cpe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpe)


def _demo(name: str, ok: bool, err: float) -> dict:
    errs = {"object_pos_max_abs_m": err, "full_state_max_abs": err}
    return {"demo": name, "replay_success_at_end": ok,
            "replay_first_success_step": 3 if ok else None,
            "max_errors": errs, "final_errors": errs}


def _write(d: Path, demos: list[dict], align: float, frame: np.ndarray) -> None:
    (d / "frames").mkdir(parents=True)
    ok = sum(x["replay_success_at_end"] for x in demos)
    (d / "data_report_lift.json").write_text(json.dumps({
        "replay": {"per_demo": demos, "replay_success_count": ok,
                   "control_steps_per_sec_median": 100.0},
        "integrity_checks": {"state_restore_exact": True},
        "obs_state_alignment_demo0": {"robot0_eef_pos": align},
        "render": {"backend": "x", "render_ms_mean": 1.0}}))
    (d / "smoke_sim_lift.json").write_text(json.dumps({"checks": {"a": True}, "all_passed": True}))
    iio.imwrite(d / "frames" / "f.png", frame)


def test_identical_dirs_report_bit_identical(tmp_path: Path) -> None:
    frame = np.full((4, 4, 3), 100, np.uint8)
    demos = [_demo("demo_0", True, 0.01), _demo("demo_1", False, 0.2)]
    _write(tmp_path / "a", demos, 0.0, frame)
    _write(tmp_path / "b", demos, 0.0, frame)
    r = cpe.compare_task(tmp_path / "a", tmp_path / "b", "lift")
    assert r["same_failed_demos"] and r["per_demo_errors_bit_identical"]
    assert r["failed_demos"]["a"] == ["demo_1"]
    assert cpe.compare_frames(tmp_path / "a", tmp_path / "b")["f.png"]["mean_abs_diff"] == 0.0


def test_differences_are_measured(tmp_path: Path) -> None:
    _write(tmp_path / "a", [_demo("demo_0", True, 0.01)], 0.0, np.zeros((4, 4, 3), np.uint8))
    _write(tmp_path / "b", [_demo("demo_0", False, 0.04)], 1e-16, np.full((4, 4, 3), 20, np.uint8))
    r = cpe.compare_task(tmp_path / "a", tmp_path / "b", "lift")
    assert not r["same_failed_demos"] and not r["per_demo_errors_bit_identical"]
    diff = r["per_demo_error_max_abs_diff"]
    assert diff["max_errors.object_pos_max_abs_m"] == pytest.approx(0.03)
    f = cpe.compare_frames(tmp_path / "a", tmp_path / "b")["f.png"]
    assert f["mean_abs_diff"] == 20.0 and f["frac_pixels_diff_gt_16"] == 1.0


def test_mismatched_demo_sets_raise(tmp_path: Path) -> None:
    frame = np.zeros((2, 2, 3), np.uint8)
    _write(tmp_path / "a", [_demo("demo_0", True, 0.0)], 0.0, frame)
    _write(tmp_path / "b", [_demo("demo_1", True, 0.0)], 0.0, frame)
    with pytest.raises(ValueError, match="different demo sets"):
        cpe.compare_task(tmp_path / "a", tmp_path / "b", "lift")
