"""The Gate A render probe runs from the Modal image's code directories alone (Session 5).

The first probe retry rendered nothing because its render check (``smoke_sim.py``) needed the
evaluation suite, which the image does not carry (KNOWN_ISSUES KI-033). This copies exactly
the image's code directories (``configs/compute/modal.yaml``) to a temporary root and runs
``scripts/render_probe.py`` from there, on this platform's default backend.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from roboforge.compute.pricing import load_modal_config
from roboforge.paths import REPO_ROOT

pytestmark = pytest.mark.sim


def test_render_probe_needs_only_the_image_code_dirs(tmp_path: Path) -> None:
    app = tmp_path / "app"
    for d in load_modal_config()["app"]["code_dirs"]:
        shutil.copytree(REPO_ROOT / d, app / d,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    assert not (app / "data" / "eval_states").exists()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ROBOFORGE_", "MUJOCO_GL", "PYOPENGL"))}
    env["PYTHONPATH"] = str(app / "src")
    out = tmp_path / "render.json"
    p = subprocess.run([sys.executable, str(app / "scripts" / "render_probe.py"), "--frames",
                        "2", "--out", str(out)], env=env, cwd=tmp_path, capture_output=True,
                       text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    res = json.loads(out.read_text())
    assert res["ok"] and set(res["cameras"]) == {"agentview", "frontview"}
    assert all(c["shape"] == [256, 256, 3] for c in res["cameras"].values())
