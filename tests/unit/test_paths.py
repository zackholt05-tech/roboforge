"""Path recording for evidence reports (``roboforge.paths.repo_relative``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from roboforge.paths import REPO_ROOT, feasibility_reports_dir, repo_relative


def test_path_inside_repo_is_relative() -> None:
    frame = feasibility_reports_dir() / "frames" / "smoke_lift_seed0_agentview.png"
    assert repo_relative(frame) == "reports/feasibility/frames/smoke_lift_seed0_agentview.png"


def test_relative_input_resolves_against_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    assert repo_relative(Path("reports") / "x.json") == "reports/x.json"


def test_repo_root_itself() -> None:
    assert repo_relative(REPO_ROOT) == "."


def test_path_outside_repo_is_absolute(tmp_path: Path) -> None:
    # A container run writes to e.g. /out; the old parents[k] arithmetic raised IndexError.
    out = tmp_path / "frames" / "x.png"
    assert repo_relative(out) == out.resolve().as_posix()
    assert repo_relative("/out/frames/x.png") == Path("/out/frames/x.png").resolve().as_posix()
