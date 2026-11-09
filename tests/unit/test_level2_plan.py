"""The Level 2 plan pins the frozen RGB diffusion reference and the suites by hash (Session 9)."""

from __future__ import annotations

import json

import yaml

from roboforge.paths import REPO_ROOT
from roboforge.provenance import sha256_file
from roboforge.sim.eval_states import read_manifest
from roboforge.train.rgb import RGBConfig, config_path

PLAN = yaml.safe_load((REPO_ROOT / "configs" / "experiments" /
                       "level2_pointcloud.yaml").read_text())


def test_rgb_reference_rows_match_the_committed_selections_and_heldout_reports() -> None:
    rows = PLAN["arms"]["rgb_diffusion"]["selections"]
    assert [r["seed"] for r in rows] == [0, 1, 2]
    for r in rows:
        assert RGBConfig.load(config_path("diffusion"), {"seed": r["seed"]}).run_id("s8") == \
            r["run_id"]
        sel = REPO_ROOT / "reports" / "session8" / "rgb_selection" / f"{r['run_id']}.json"
        assert sha256_file(sel) == r["selection_sha256"]
        s = json.loads(sel.read_text())["selected"]
        assert (s["step"], s["sha256"]) == (r["step"], r["checkpoint_sha256"])
        rep = REPO_ROOT / "reports" / "session8" / "rgb_heldout" / f"{r['run_id']}.json"
        assert sha256_file(rep) == r["heldout_report_sha256"]


def test_suite_hashes_match_the_committed_manifests() -> None:
    s = PLAN["suites"]
    for key in ("selection", "placement_base", "placement_ood"):
        assert read_manifest("lift", s[key]["name"])["content_sha256"] == \
            s[key]["manifest_content_sha256"], key
    # The held-out schedule is pinned by the hash the RGB freeze and Level 1 already fixed;
    # its manifest is not opened here.
    freeze = (REPO_ROOT / "docs" / "RGB_EXPERIMENT_FREEZE.md").read_text()
    assert s["heldout"]["manifest_content_sha256"] in freeze


def test_plan_arms_and_analysis() -> None:
    assert PLAN["arms"]["pc_diffusion"]["seeds"] == [0, 1, 2]
    assert PLAN["arms"]["pc_diffusion"]["config"] == \
        "configs/models/pointcloud/diffusion_dp3_lift.yaml"
    b = PLAN["analysis"]["bootstrap"]
    assert (b["B"], b["seed"]) == (10000, 20261106)
    assert PLAN["evaluation"] == {"policy_seed": 0, "repeat": 0, "device": "cpu",
                                  "torch_threads": 1, "platform": "canonical"}
