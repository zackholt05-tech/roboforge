"""``scripts/verify_data.py`` honours ``--out-dir`` for both the report and rendered frames.

Regression test: frames were always written to ``reports/feasibility/frames``, so a
container run with ``--out-dir /out`` lost them, and a report path computed with
``out_dir.parents[2]`` raised IndexError for shallow output directories.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from roboforge.paths import REPO_ROOT

pytestmark = [pytest.mark.data, pytest.mark.render]


def test_report_and_frames_go_to_out_dir(lift_lowdim_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "verify_data.py"), "--tasks", "lift",
           "--replay-demos", "1", "--render", "--out-dir", str(out)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]

    report = json.loads((out / "data_report_lift.json").read_text())
    frames = report["render"]["frames"]
    assert frames, "no frames recorded"
    for f in frames:
        path = Path(f["path"])
        assert path.is_absolute(), f"external out-dir must be recorded absolute: {path}"
        assert path.parent == (out / "frames").resolve()
        assert path.is_file()
