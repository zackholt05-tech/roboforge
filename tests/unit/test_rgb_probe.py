"""Session 8 Gate D RGB probe: plan bound, approval A-02, attempt cap and the CPU job path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboforge.compute import rgb_bench
from roboforge.compute.approval import ApprovalError, authorize, read_approvals
from roboforge.compute.pricing import load_modal_config
from roboforge.compute.rgb_probe import JOB_MARGIN_S, rgb_probe_job, rgb_probe_plan, sections


def _tiny_plan(deadline_s: int = 10_000) -> dict:
    plan = rgb_probe_plan(events=[])
    plan["deadline_s"] = deadline_s
    plan["rgb_probe"]["sections"] = [
        {"name": "train:default:compact:84:bc_chunk:b2", "kind": "train", "trunk": "compact",
         "size": 84, "head": "bc_chunk", "batch": 2, "mode": "default", "steps": 1,
         "warmup": 0},
        {"name": "repeat:default:compact:84:diffusion:b2", "kind": "repeat",
         "trunk": "compact", "size": 84, "head": "diffusion", "batch": 2, "mode": "default",
         "steps": 2}]
    return plan


def test_plan_upper_bound_fits_the_cap_even_at_the_region_maximum() -> None:
    cfg = load_modal_config()
    plan = rgb_probe_plan(events=[])
    assert plan["function"] == "rgb_probe" and plan["resources"]["gpu"] == "L4"
    assert plan["run_id"] is None and plan["mode"] == "probe"
    assert plan["deadline_s"] == plan["timeout_s"] - JOB_MARGIN_S
    total = plan["estimate"]["total_usd"]
    assert total <= 1.0
    assert total * cfg["pricing"]["multipliers"]["region_max"] <= 1.0
    names = [s["name"] for s in plan["rgb_probe"]["sections"]]
    assert len(names) == len(set(names)) == 22
    assert names[0].startswith("train:default:resnet18:84:")
    assert any(n.startswith("repeat:deterministic:") for n in names)


def test_approval_a02_covered_only_the_rgb_probe_and_is_now_closed() -> None:
    a = read_approvals()["A-02"]
    assert (a.status, a.functions, a.cap_usd) == ("exhausted", ("rgb_probe",), 1.0)
    est = rgb_probe_plan(events=[])["estimate"]["total_usd"]
    assert est <= a.cap_usd
    with pytest.raises(ApprovalError, match="exhausted"):
        authorize("A-02", "rgb_probe", est, [])


def test_job_refuses_once_the_attempt_cap_is_reached(tmp_path: Path) -> None:
    plan = _tiny_plan()
    d = tmp_path / plan["launch_id"]
    d.mkdir()
    for n in range(plan["policy"]["max_attempts_per_launch"]):
        (d / f"attempt_{n}.json").write_text("{}")
    res = rgb_probe_job(plan, attempts_root=tmp_path, device="cpu")
    assert res["status"] == "refused" and "sections" not in res
    assert len(list(d.glob("attempt_*.json"))) == plan["policy"]["max_attempts_per_launch"]


def test_job_runs_sections_records_attempts_and_commits(tmp_path: Path) -> None:
    plan = _tiny_plan()
    commits: list[int] = []
    res = rgb_probe_job(plan, attempts_root=tmp_path, device="cpu",
                        commit=lambda: commits.append(1))
    assert res["status"] == "complete", res.get("failed_sections")
    assert res["attempt"] == 0 and res["container_wall_s"] > 0
    rep = res["sections"]["repeat:default:compact:84:diffusion:b2"]["value"]
    assert rep["bitwise_equal"] and rep["step_def"]["ema"]
    saved = json.loads((tmp_path / plan["launch_id"] / "attempt_0.json").read_text())
    assert saved["status"] == "complete" and len(commits) >= 2
    again = rgb_probe_job(plan, attempts_root=tmp_path, device="cpu")
    assert again["attempt"] == 1


def test_job_past_its_deadline_skips_sections(tmp_path: Path) -> None:
    res = rgb_probe_job(_tiny_plan(deadline_s=-1), attempts_root=tmp_path, device="cpu")
    assert res["status"] == "partial" and res["sections"] == {}
    assert len(res["skipped_sections"]) == 2


def test_bench_step_definition_and_cpu_timing() -> None:
    r = rgb_bench.train_steps("compact", 84, "diffusion", 2, 1, 0, "cpu")
    assert r["step_def"]["ema"] and r["step_def"]["grad_norm"]
    assert not r["step_def"]["h2d_copy_in_step"] and r["peak_memory_mib"] is None
    assert r["flags"]["mode"] == "default" and r["median_s_per_step"] > 0
    b = rgb_bench.train_steps("compact", 84, "bc_chunk", 2, 1, 0, "cpu")
    assert not b["step_def"]["ema"]


def test_sections_put_the_decision_relevant_configs_first() -> None:
    names = [s["name"] for s in sections(30, 5, 20)]
    first_compact = next(i for i, n in enumerate(names) if ":compact:" in n)
    assert all(":resnet18:" in n for n in names[:first_compact])
