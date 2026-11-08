"""``scripts/run_suite.py`` reads of files that running jobs write (KNOWN_ISSUES KI-036):
a half-written summary or manifest is "not there yet" while its job runs, and a manual
decision (``SuiteError``) when no job is writing it; a running step is never read."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from roboforge.paths import REPO_ROOT


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module
    spec.loader.exec_module(mod)
    return mod


rs = _load("run_suite")
RID = "s7_x_can_seed0_00000000"
SUMMARY = f"{RID}__val_step0005000__linux-aarch64-container-1f685638f28c.json"


def _suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, running: list[str]):
    monkeypatch.setattr(rs, "REPO_ROOT", tmp_path)
    s = rs.Suite.__new__(rs.Suite)
    s.reports = tmp_path / "reports"
    s.jobs = [SimpleNamespace(key=k) for k in running]
    (s.reports / "validation").mkdir(parents=True)
    (tmp_path / "runs" / RID).mkdir(parents=True)
    return s


def test_half_written_summary_of_a_running_evaluation_is_not_there_yet(tmp_path, monkeypatch):
    s = _suite(tmp_path, monkeypatch, running=[f"{RID}@5000"])
    (s.reports / "validation" / SUMMARY).write_text('{"eligible_for')
    assert s.eval_summary(RID, 5000) is None


def test_unparseable_summary_without_a_running_job_needs_a_decision(tmp_path, monkeypatch):
    s = _suite(tmp_path, monkeypatch, running=[f"{RID}@10000"])
    (s.reports / "validation" / SUMMARY).write_text("")
    with pytest.raises(rs.SuiteError, match="no running job"):
        s.eval_summary(RID, 5000)


def test_half_written_manifest_of_a_running_training_is_not_there_yet(tmp_path, monkeypatch):
    s = _suite(tmp_path, monkeypatch, running=[RID])
    (tmp_path / "runs" / RID / "manifest.json").write_text('{"status": "comp')
    assert s.manifest(RID) is None
    s.jobs = []
    with pytest.raises(rs.SuiteError):
        s.manifest(RID)


def test_complete_files_are_read(tmp_path, monkeypatch):
    s = _suite(tmp_path, monkeypatch, running=[])
    rec = {"eligible_for_selection": True, "partial": False, "n_episodes": 50, "successes": 7}
    (s.reports / "validation" / SUMMARY).write_text(json.dumps(rec))
    (tmp_path / "runs" / RID / "manifest.json").write_text('{"status": "complete"}')
    assert s.eval_summary(RID, 5000) == rec
    assert s.manifest(RID) == {"status": "complete"}
