"""RGB study launches (A-03, now closed): plan bound, approval, attempt cap and run-ID reuse."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboforge.compute.approval import ApprovalError, authorize, read_approvals
from roboforge.compute.rgb_train import run_segment, train_plan
from roboforge.paths import REPO_ROOT
from roboforge.train.rgb import RGBConfig, config_path

INSTALLED = (REPO_ROOT / "data" / "manifests" / "rgb" / "lift_rgb84_v1.json").is_file()


@pytest.mark.skipif(not INSTALLED, reason="RGB dataset not installed")
@pytest.mark.parametrize("family", ["bc_chunk", "diffusion"])
def test_plan_fit_one_launch_under_a03_which_is_now_closed(family: str,
                                                        lift_processed_path: Path) -> None:
    plan = train_plan(family, 0, "s8", None, events=[])
    assert plan["function"] == "train_gpu" and plan["timeout_s"] == 3600
    assert plan["resources"]["gpu"] == "L4" and plan["policy"]["max_attempts_per_launch"] == 4
    assert plan["run_id"].startswith(f"s8_rgb_{family}_resnet18_lift_seed0_")
    m = json.loads((REPO_ROOT / "data" / "manifests" / "rgb" / "lift_rgb84_v1.json")
                   .read_text())
    assert plan["data"]["rgb_content_sha256"] == m["content_sha256"]
    est = plan["estimate"]["total_usd"]
    assert est < 4.6
    a = read_approvals()["A-03"]
    assert (a.status, a.functions, a.cap_usd) == ("exhausted", ("train_gpu",), 15.0)
    assert est < a.cap_usd / 3  # a launch's 4-attempt bound fitted the cap with room to spare
    with pytest.raises(ApprovalError, match="exhausted"):
        authorize("A-03", "train_gpu", est, [])


def test_seed_changes_the_run_id_but_not_the_rest_of_the_config() -> None:
    a = RGBConfig.load(config_path("diffusion"), {"seed": 0})
    b = RGBConfig.load(config_path("diffusion"), {"seed": 1})
    assert a.run_id("s8") != b.run_id("s8")
    assert {k for k in a.raw if a.raw[k] != b.raw[k]} == {"seed"}


def test_segment_refuses_past_the_attempt_cap_and_new_run_reuse(tmp_path: Path) -> None:
    rid = RGBConfig.load(config_path("bc_chunk"), {"seed": 0}).run_id("s8t")
    d = tmp_path / rid
    d.mkdir()
    (d / "segments.jsonl").write_text("".join(json.dumps({"launch_id": "L1"}) + "\n"
                                              for _ in range(4)))
    kw = {"family": "bc_chunk", "seed": 0, "prefix": "s8t", "overrides": {},
          "runs_root": tmp_path, "device": "cpu", "deadline_s": None}
    res = run_segment(**kw, launch_id="L1", mode="new", max_attempts=4)
    assert res["status"] == "refused" and res["attempt"] == 4
    with pytest.raises(FileExistsError):
        run_segment(**kw, launch_id="L2", mode="new", max_attempts=4)
    with pytest.raises(FileNotFoundError):
        run_segment(**kw, launch_id="L3", mode="resume", max_attempts=4)
