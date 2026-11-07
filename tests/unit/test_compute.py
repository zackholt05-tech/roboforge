"""Cost arithmetic, approval gate, cost ledger, launch rules and retrieval checks (Session 5).

No test contacts Modal: the SDK is not installed in the project venv (DECISIONS D-045), and a
test asserts that no RoboForge module imports it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from roboforge.compute import cost_log
from roboforge.compute.approval import ApprovalError, authorize, read_approvals
from roboforge.compute.plan import _checks, authorize_plan, record_result
from roboforge.compute.pricing import (
    Resources,
    cost_usd,
    launch_upper_bound,
    load_modal_config,
    storage_month_usd,
)
from roboforge.compute.retrieve import RetrievalError, install, verify_run_dir
from roboforge.paths import REPO_ROOT
from roboforge.provenance import sha256_file

CFG = load_modal_config()
PRICING = CFG["pricing"]
LEDGER_TEXT = """
| ID | Date | Gate | Functions | Cap (USD) | Status | Evidence |
|---|---|---|---|---|---|---|
| A-01 | 2026-11-06 | Gate A | gpu_probe, train_cpu | 2.00 | granted | "yes" |
| A-02 | 2026-11-06 | Gate C | train_cpu | 30.00 | revoked | withdrawn |
"""


def test_list_price_arithmetic() -> None:
    r = Resources(seconds=3600, cpu=2.0, memory_gib=4.0, gpu="L4")
    c = cost_usd(r, PRICING)
    assert c["gpu_usd"] == pytest.approx(0.000222 * 3600)
    assert c["cpu_usd"] == pytest.approx(0.0000131 * 2 * 3600)
    assert c["memory_usd"] == pytest.approx(0.00000222 * 4 * 3600)
    assert c["total_usd"] == pytest.approx(c["gpu_usd"] + c["cpu_usd"] + c["memory_usd"])
    assert cost_usd(r, PRICING, multiplier=3.0)["total_usd"] == pytest.approx(
        3 * c["total_usd"])
    with pytest.raises(ValueError):
        cost_usd(Resources(1, 0.1, 1.0), PRICING)  # below Modal's 0.125-core minimum
    with pytest.raises(KeyError):
        cost_usd(Resources(1, 1.0, 1.0, gpu="B9000"), PRICING)
    assert storage_month_usd(10.0, PRICING) == 0.0  # inside the free 1 TiB
    assert storage_month_usd(1034.0, PRICING) == pytest.approx(0.9)


def test_upper_bound_covers_timeout_keepalive_and_build() -> None:
    ub = launch_upper_bound(CFG, "gpu_probe", 900)
    f = CFG["functions"]["gpu_probe"]
    held = 900 + CFG["policy"]["scaledown_window_s"]
    job = cost_usd(Resources(held, f["cpu"], f["memory_mib"] / 1024, f["gpu"]), PRICING)
    assert ub["job"]["total_usd"] == pytest.approx(job["total_usd"])
    assert ub["total_usd"] == pytest.approx(job["total_usd"] + ub["build_allowance"]["total_usd"])
    assert "estimate" in ub["kind"]


def test_launch_policy_is_single_container_no_retry_one_gpu() -> None:
    pol = CFG["policy"]
    assert pol["max_containers"] == 1 and pol["retries"] == 0
    assert pol["scaledown_window_s"] == 2 and pol["deadline_margin_s"] > 0
    for name, f in CFG["functions"].items():
        assert f["gpu"] in (None, "L4", "T4"), name  # a single small GPU, never multi-GPU
        assert ":" not in str(f["gpu"]), "no GPU count suffix (multi-GPU) allowed"
        assert 60 <= f["default_timeout_s"] <= f["max_timeout_s"] <= 6 * 3600
    src = (REPO_ROOT / "infra" / "modal_app.py").read_text()
    for needle in ('"max_containers": POL["max_containers"]', '"retries": POL["retries"]',
                   '"single_use_containers": True', 'with_options(timeout=plan["timeout_s"])'):
        assert needle in src


def test_no_roboforge_module_imports_the_modal_sdk() -> None:
    pat = re.compile(r"^\s*(import modal|from modal\b)", re.MULTILINE)
    offenders = [p for p in (REPO_ROOT / "src").rglob("*.py") if pat.search(p.read_text())]
    assert offenders == []


def test_approval_rows_and_gate(tmp_path: Path) -> None:
    doc = tmp_path / "ledger.md"
    doc.write_text(LEDGER_TEXT)
    a = read_approvals(doc)
    assert a["A-01"].functions == ("gpu_probe", "train_cpu") and a["A-01"].cap_usd == 2.0
    ok = authorize("A-01", "gpu_probe", 0.5, [], doc)
    assert ok["remaining_after_usd"] == pytest.approx(1.5)
    for bad in [(None, "gpu_probe", 0.1), ("A-09", "gpu_probe", 0.1),
                ("A-02", "train_cpu", 0.1), ("A-01", "eval_cpu", 0.1),
                ("A-01", "gpu_probe", 2.01)]:
        with pytest.raises(ApprovalError):
            authorize(*bad, events=[], path=doc)
    events = [{"event": "launch", "launch_id": "x", "approval_id": "A-01",
               "estimate_upper_usd": 1.8, "utc": "2026-11-06T00:00:00+00:00"}]
    with pytest.raises(ApprovalError, match="exceeds the cap"):
        authorize("A-01", "gpu_probe", 0.3, events, doc)  # 1.8 committed + 0.3 > 2.0
    events.append({"event": "result", "launch_id": "x", "measured_cost_usd": 0.2,
                   "utc": "2026-11-06T01:00:00+00:00"})
    assert authorize("A-01", "gpu_probe", 0.3, events, doc)["committed_before_usd"] == 0.2


def test_cost_ledger_bases_and_sums(tmp_path: Path) -> None:
    led = tmp_path / "l.jsonl"
    cost_log.append_event("launch", {"launch_id": "a", "approval_id": "A-01", "run_id": "r1",
                                     "estimate_upper_usd": 1.0}, led)
    cost_log.append_event("launch", {"launch_id": "b", "approval_id": "A-01", "run_id": "r2",
                                     "estimate_upper_usd": 2.0}, led)
    cost_log.append_event("result", {"launch_id": "a", "measured_cost_usd": 0.25}, led)
    cost_log.append_event("billing", {"launch_id": "b", "billed_usd": 0.5,
                                      "source": "dashboard"}, led)
    ev = cost_log.read_events(led)
    assert cost_log.launch_cost("a", ev)["basis"].startswith("measured")
    assert cost_log.launch_cost("b", ev) == {"usd": 0.5, "basis": "billed"}
    assert cost_log.committed_usd(ev, approval_id="A-01") == pytest.approx(0.75)
    month = ev[0]["utc"][:7]
    assert cost_log.committed_usd(ev, month=month) == pytest.approx(0.75)
    assert cost_log.committed_usd(ev, month="1999-01") == 0.0
    assert cost_log.launched_run_ids(ev) == {"r1", "r2"}
    with pytest.raises(ValueError):
        cost_log.append_event("spend", {}, led)


def test_launch_rules_refuse_reused_run_ids(monkeypatch) -> None:
    monkeypatch.setattr("roboforge.compute.plan.git_revision",
                        lambda: {"sha": "abc", "dirty": False})
    ev = [{"event": "launch", "launch_id": "a", "run_id": "r1"}]
    assert _checks("new", "r_unused_xyz", ev) == []
    assert any("already exists" in b for b in _checks("new", "r1", ev))
    assert _checks("resume", "r1", ev) == []
    assert any("never launched" in b for b in _checks("resume", "r_unused_xyz", ev))
    monkeypatch.setattr("roboforge.compute.plan.git_revision",
                        lambda: {"sha": "abc", "dirty": True})
    assert any("dirty" in b for b in _checks("new", "r_unused_xyz", ev))


def test_authorize_plan_logs_once_and_record_result_prices_measured_time(
        tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("roboforge.compute.plan.git_revision",
                        lambda: {"sha": "abc", "dirty": False})
    doc = tmp_path / "ledger.md"
    doc.write_text(LEDGER_TEXT)
    monkeypatch.setattr("roboforge.compute.approval.LEDGER_DOC", doc)
    monkeypatch.setattr("roboforge.compute.plan.authorize",
                        lambda a, f, e, ev: authorize(a, f, e, ev, doc))
    led = tmp_path / "cost.jsonl"
    plan = {"function": "gpu_probe", "launch_id": "L1", "run_id": None, "mode": "probe",
            "timeout_s": 900, "deadline_s": 720, "git": {"sha": "abc", "dirty": False},
            "resources": {"seconds": 900, "cpu": 4.0, "memory_gib": 16.0, "gpu": "L4",
                          "count": 1}, "policy": CFG["policy"],
            "estimate": launch_upper_bound(CFG, "gpu_probe", 900),
            "image": {"uv_lock_sha256": "x"}}
    ev = authorize_plan(plan, "A-01", led)
    assert ev["event"] == "launch" and ev["approval_id"] == "A-01"
    with pytest.raises(ApprovalError, match="already authorized"):
        authorize_plan(plan, "A-01", led)
    res = record_result(plan, {"status": "complete", "container_wall_s": 100.0}, led)
    expect = cost_usd(Resources(102.0, 4.0, 16.0, "L4"), PRICING)["total_usd"]
    assert res["measured_cost_usd"] == pytest.approx(expect)


def _fake_run(root: Path, rid: str) -> Path:
    d = root / rid
    (d / "checkpoints").mkdir(parents=True)
    cfg = {"name": "x"}
    (d / "config.json").write_text(json.dumps(cfg))
    cks, idx = [], []
    for s in (10, 20):
        f = d / "checkpoints" / f"step_{s:07d}.pt"
        f.write_bytes(f"ckpt{s}".encode())
        c = {"step": s, "path": f"runs/{rid}/checkpoints/{f.name}", "sha256": sha256_file(f)}
        cks.append(c)
        idx.append(json.dumps({**c, "segment": 0, "saved_utc": "t"}))
    (d / "checkpoints.jsonl").write_text("\n".join(idx) + "\n")
    (d / "metrics.jsonl").write_text('{"step": 10}\n')
    m = {"run_id": rid, "status": "complete", "reload_check": {"passed": True}, "config": cfg,
         "checkpoints": cks, "resume_checkpoints": [],
         "metrics_sha256": sha256_file(d / "metrics.jsonl")}
    (d / "manifest.json").write_text(json.dumps(m))
    return d


def test_retrieval_verifies_hashes_and_never_overwrites(tmp_path: Path) -> None:
    staged = _fake_run(tmp_path / "staging", "r1")
    assert verify_run_dir(staged, "r1")["n_checkpoints"] == 2
    out = install(staged, "r1", tmp_path / "runs")
    assert out["status"] == "PASS" and (tmp_path / "runs" / "r1" / "manifest.json").is_file()
    with pytest.raises(RetrievalError, match="never overwritten"):
        install(staged, "r1", tmp_path / "runs")
    with pytest.raises(RetrievalError):
        verify_run_dir(staged, "r2")  # wrong run ID
    (staged / "checkpoints" / "step_0000020.pt").write_bytes(b"tampered")
    with pytest.raises(RetrievalError, match="SHA-256"):
        verify_run_dir(staged, "r1")
    staged2 = _fake_run(tmp_path / "staging2", "r1")
    (staged2 / "metrics.jsonl").write_text('{"step": 11}\n')
    with pytest.raises(RetrievalError, match="metrics"):
        verify_run_dir(staged2, "r1")
    staged3 = _fake_run(tmp_path / "staging3", "r1")
    (staged3 / "manifest.json").unlink()
    with pytest.raises(RetrievalError, match="did not finish"):
        verify_run_dir(staged3, "r1")


def test_suite_counts_match_the_protocol_budget() -> None:
    from roboforge.compute.estimate import suite_counts

    full = suite_counts("max", 100_000, 10_000)
    # 4 families x 2 tasks; 4 configurations at seed 0 + seeds 1-2 of the chosen one.
    assert full["training_runs"] == 4 * 2 * 6 == 48
    assert full["checkpoints_per_run"] == 10
    assert full["validation_evaluations"] == 480 and full["test_evaluations"] == 24
    none = suite_counts("none", 50_000, 10_000)
    assert none["training_runs"] == 24 and none["validation_evaluations"] == 120


def test_job_estimate_cli_prices_the_requested_gpu(capsys) -> None:
    from roboforge.compute.estimate import main

    assert main(["job", "--gpu", "T4", "--max-minutes", "15"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["job"]["resources"]["gpu"] == "T4"
    assert out["job"]["gpu_usd"] == pytest.approx(0.000164 * (900 + 2))
