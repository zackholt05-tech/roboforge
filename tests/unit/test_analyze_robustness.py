"""Robustness analysis loader on summaries in the writer's layout (scripts/analyze_robustness.py,
KI-038): the checkpoint step lives under ``checkpoint.step`` and ``inference_variant`` is
present only for a variant (``roboforge.eval.validation``)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from roboforge.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location("analyze_robustness",
                                              REPO_ROOT / "scripts" / "analyze_robustness.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


def _write(d: Path, name: str, *, step: int, suite: str, variant: str | None = None,
           canonical: bool = True) -> None:
    s = {"train_run_id": "run_a", "checkpoint": {"step": step, "sha256": "0" * 64},
         "suite": suite, "canonical_platform": canonical,
         **({} if variant is None else {"inference_variant": variant})}
    (d / f"{name}.json").write_text(json.dumps(s))


def test_evals_key_on_checkpoint_step_and_variant(tmp_path: Path) -> None:
    _write(tmp_path, "base", step=65000, suite="robustness_base")
    _write(tmp_path, "ema", step=65000, suite="robustness_base", variant="trained-weights")
    _write(tmp_path, "ood", step=65000, suite="ood_arm_offset")
    ev = ar.Evals(tmp_path)
    assert set(ev.by_key) == {("run_a", 65000, "robustness_base", None),
                              ("run_a", 65000, "robustness_base", "trained-weights"),
                              ("run_a", 65000, "ood_arm_offset", None)}
    assert ev.get("run_a", 65000, "robustness_base")[1].name == "base.json"
    assert ev.get("run_a", 65000, "robustness_base", "trained-weights")[1].name == "ema.json"
    assert ev.get("run_a", 60000, "robustness_base") is None


def test_evals_refuse_duplicates_and_non_canonical(tmp_path: Path) -> None:
    _write(tmp_path, "a", step=5000, suite="robustness_base")
    _write(tmp_path, "b", step=5000, suite="robustness_base")
    with pytest.raises(ValueError, match="two summaries"):
        ar.Evals(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _write(other, "native", step=5000, suite="robustness_base", canonical=False)
    with pytest.raises(ValueError, match="not canonical"):
        ar.Evals(other)


def test_main_refuses_to_overwrite_an_existing_analysis(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Run once (audit B-12): an existing analysis.json stops the script before any read."""
    (tmp_path / "analysis.json").write_text("{}")
    monkeypatch.setattr(ar, "evidence_reports_dir", lambda *parts: tmp_path)
    with pytest.raises(SystemExit, match="analysed once"):
        ar.main(["--config", "lift=configs/experiments/level1_core_lift.yaml"])
    assert (tmp_path / "analysis.json").read_text() == "{}"
