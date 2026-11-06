"""Run the three.js clip check (``check_clip.mjs``) from pytest when node + three are present."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
CLIP = HERE.parents[1] / "reports" / "session2" / "web_diagnostic" / "clip_lift_demo_0.json.gz"

pytestmark = pytest.mark.web


def test_exported_clip_passes_the_real_threejs_checks(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    if not (HERE / "node_modules" / "three").exists():
        pytest.skip("run `npm ci` in tests/web_contract first (make web-contract)")
    if not CLIP.exists():
        pytest.skip(f"{CLIP} not exported (make export-clip)")
    out = tmp_path / "three_check.json"
    proc = subprocess.run(["node", str(HERE / "check_clip.mjs"), str(CLIP), str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(out.read_text())
    assert all(result["checks"].values())
    assert result["three_revision"] == "187"
