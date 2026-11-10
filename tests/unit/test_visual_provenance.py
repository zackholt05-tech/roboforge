"""``docs/VISUAL_PROVENANCE.md`` is generated from the export index, one row per clip."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

from roboforge.paths import REPO_ROOT

_spec = importlib.util.spec_from_file_location("visual_provenance",
                                               REPO_ROOT / "scripts" / "visual_provenance.py")
vp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vp)


def _write(out: Path, ver: dict | None) -> None:
    (out / "clips").mkdir(parents=True)
    clips = []
    for cid, ck, has in (("lift-diffusion_unet-s0-x", {"sha256": "a" * 64, "step": 100000},
                          ["reference_frames"]),
                         ("lift-random_uniform-s0-x", {"sha256": None, "step": None}, [])):
        fam = cid.split("-")[1]
        with gzip.open(out / "clips" / f"{cid}.json.gz", "wt") as fh:
            json.dump({"policy": {"name": fam}, "provenance": {"simulator": {
                "packages": {"mujoco": "3.2.7", "robosuite": "1.5.1"}}}}, fh)
        clips.append({"id": cid, "roles": ["lab"], "seed": 0, "state_id": "lift_final_test_0000",
                      "partition": "eval_final_test", "eval_run_id": f"run_{fam}",
                      "checkpoint": ck, "has": has})
    (out / "index.json").write_text(json.dumps({
        "platform": {"platform_id": "linux-aarch64-container-1f685638f28c"},
        "git": {"sha": "b" * 40, "dirty": False}, "wall_s": 3600.0, "clips": clips,
        "files": {"clips/x": "c" * 64}}))
    if ver is not None:
        (out / "viewer_verification.json").write_text(json.dumps(ver))


def test_one_row_per_clip_and_non_learned_has_no_checkpoint(tmp_path: Path) -> None:
    _write(tmp_path, None)
    text = vp.build(tmp_path)
    rows = [ln for ln in text.splitlines() if ln.startswith("| `lift-")]
    assert len(rows) == 2
    assert "100000, `aaaaaaaaaaaa`" in rows[0] and "reference_frames" in rows[0]
    assert "none (non-learned)" in rows[1]
    assert "MuJoCo 3.2.7, robosuite 1.5.1" in rows[0]
    assert "Not run yet" in text


def test_verification_summary_reports_the_range_and_verdict(tmp_path: Path) -> None:
    frames = [{"robot_iou": 0.9998, "object_iou": 1.0, "robot_centroid_px": 0.03},
              {"robot_iou": 1.0, "object_iou": None, "robot_centroid_px": 0.0}]
    _write(tmp_path, {"frames": frames, "pass": True, "threshold": {"robot_iou_min": 0.95},
                      "method": "silhouettes"})
    text = vp.build(tmp_path)
    assert "2 reference frames, overall **pass**" in text
    assert "Robot IoU 0.9998–1.0000" in text and "at most 0.03 px" in text
