"""The X3 validation driver does what ``run_suite.py`` would have done (DECISIONS D-112).

- Its evaluation argv equals every evaluation launch the X3 suite logged, and its view of a
  training process matches every training launch the suite logged.
- Its eligibility and training-status rules (one eligible 50-start summary per checkpoint;
  a run without a final manifest is waited for only while it trains).
- Its finish path, on the real BC-MLP seed 3 evidence (selected by the suite before it
  stopped) and on synthetic runs, through to the X3 gate in ``x3_jobs``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from roboforge.paths import REPO_ROOT
from roboforge.replication import x3_jobs
from roboforge.replication import x3_validation as xv

SUITE_EVENTS = REPO_ROOT / "logs/session14/suite/x3_can/events.jsonl"


def _launches(kind: str) -> list[dict]:
    if not SUITE_EVENTS.exists():
        pytest.skip("the X3 suite's events are not present")
    evs = [json.loads(line) for line in SUITE_EVENTS.read_text().splitlines()]
    out = [e for e in evs if e.get("event") == "launch" and e.get("kind") == kind]
    assert out
    return out


def test_eval_argv_equals_the_suite_launches() -> None:
    for e in _launches("eval"):
        rid, step = e["key"].split("@")
        assert xv.eval_argv(rid, int(step), "reports/session14") == e["argv"]


def test_training_processes_are_recognised_as_the_suite_launched_them() -> None:
    specs = {sp["run_id"]: sp for sp in xv.run_specs()}
    launches = _launches("train")
    assert sorted(e["key"] for e in launches) == sorted(specs)
    for e in launches:
        sp = specs[e["key"]]
        assert e["argv"][1:] == sp["train_argv"]
        assert xv.training_alive(sp["train_argv"], [" ".join(e["argv"])])
        others = [" ".join(o["argv"]) for o in launches if o["key"] != e["key"]]
        assert not xv.training_alive(sp["train_argv"], others)


def test_select_argv_is_the_suites() -> None:
    assert xv.select_argv("py", "r", "reports/session14") == [
        "py", "scripts/select_checkpoint.py", "--train-run", "runs/r", "--reports-dir",
        "reports/session14"]
    src = (REPO_ROOT / "scripts/run_suite.py").read_text()
    assert ('argv = [sys.executable, "scripts/select_checkpoint.py", "--train-run", '
            'f"runs/{rid}",\n                "--reports-dir", repo_relative(self.reports)]') in src


def _summary(path: Path, **kw) -> None:
    s = {"eligible_for_selection": True, "partial": False, "n_episodes": 50, **kw}
    path.write_text(json.dumps(s))


def test_eval_summary_rules(tmp_path: Path) -> None:
    v = tmp_path / "validation"
    v.mkdir()
    assert xv.eval_summary(tmp_path, "r", 5000, busy=False) is None
    _summary(v / "r__val_step0005000__p.json", partial=True)
    _summary(v / "r__val_step0005000__q.json", n_episodes=49)
    assert xv.eval_summary(tmp_path, "r", 5000, busy=False) is None
    _summary(v / "r__val_step0005000__z.json")
    assert xv.eval_summary(tmp_path, "r", 5000, busy=False)["n_episodes"] == 50
    _summary(v / "r__val_step0005000__zz.json")
    with pytest.raises(xv.ValidationError):
        xv.eval_summary(tmp_path, "r", 5000, busy=False)
    (v / "r__val_step0010000__p.json").write_text("{\"half")
    assert xv.eval_summary(tmp_path, "r", 10000, busy=True) is None
    with pytest.raises(xv.ValidationError):
        xv.eval_summary(tmp_path, "r", 10000, busy=False)


def test_training_status(tmp_path: Path) -> None:
    assert xv.training_status(tmp_path, "r", alive=True) == ("training", None)
    assert xv.training_status(tmp_path, "r", alive=False) == ("missing", None)
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "manifest.json").write_text("{\"half")
    assert xv.training_status(tmp_path, "r", alive=True)[0] == "training"
    with pytest.raises(xv.ValidationError):
        xv.training_status(tmp_path, "r", alive=False)
    for st in ("complete", "failed"):
        (tmp_path / "r" / "manifest.json").write_text(json.dumps({"status": st}))
        assert xv.training_status(tmp_path, "r", alive=False)[0] == st


def test_finish_path_on_the_bc_mlp_seed3_evidence() -> None:
    (sp,) = [s for s in xv.run_specs() if s["family"] == "bc_mlp" and s["seed"] == 3]
    reports = REPO_ROOT / "reports/session14"
    if not (REPO_ROOT / "runs" / sp["run_id"] / "manifest.json").exists():
        pytest.skip("the X3 training runs are not present")
    (row,) = xv.survey([sp], reports, REPO_ROOT / "runs", set(), [])
    assert (row["status"], row["missing"], row["selected"]) == ("complete", [], True)
    assert len(row["steps"]) == 20
    rec = xv.summary_record([row], reports)
    sel = json.loads(xv.selection_path(reports, sp["run_id"]).read_text())
    (out,) = rec["runs"]
    assert out["selection"] == {k: sel["chosen"][k] for k in xv.SELECTION_KEYS}
    assert out["rule"] == x3_jobs.SELECTION_RULE
    assert rec["failed"] == [] and rec["schema"] == xv.SCHEMA


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fake_suite(root: Path, failed_rid: str) -> tuple[Path, list[dict]]:
    """12 synthetic X3 runs under ``root``: one failed in training, the rest selected."""
    reports, runs = root / "reports", root / "runs"
    (reports / "selection").mkdir(parents=True)
    (reports / "validation").mkdir()
    rows = []
    for sp in xv.run_specs():
        rid = sp["run_id"]
        (runs / rid).mkdir(parents=True)
        status = "failed" if rid == failed_rid else "complete"
        ck = [{"step": s, "sha256": _sha(f"{rid}{s}".encode())} for s in (5000, 10000)]
        m = {"run_id": rid, "status": status, "checkpoints": ck if status == "complete" else []}
        (runs / rid / "manifest.json").write_text(json.dumps(m))
        row = {"family": sp["family"], "seed": sp["seed"], "run_id": rid, "status": status,
               "selected": status == "complete", "missing": [], "manifest": m}
        if status == "complete":
            srcs = []
            for c in ck:
                f = reports / "validation" / f"{rid}__val_step{c['step']:07d}__p.json"
                f.write_text(json.dumps({"step": c["step"]}))
                srcs.append({"path": str(f.relative_to(root)), "sha256": _sha(f.read_bytes())})
            sel = {"train_run_id": rid, "rule": x3_jobs.SELECTION_RULE, "sources": srcs,
                   "chosen": {"step": 10000, "sha256": ck[1]["sha256"], "successes": 40,
                              "n_episodes": 50, "first_success_step_mean": 120.0,
                              "eval_run_id": f"{rid}__val_step0010000__p"}}
            xv.selection_path(reports, rid).write_text(json.dumps(sel))
        rows.append(row)
    return reports, rows


def test_summary_and_gate_on_synthetic_runs(tmp_path: Path, monkeypatch) -> None:
    specs = xv.run_specs()
    failed = next(sp for sp in specs if sp["family"] == "bc_rnn" and sp["seed"] == 4)
    reports, rows = _fake_suite(tmp_path, failed["run_id"])
    rec = xv.summary_record(rows, reports, root=tmp_path)
    assert rec["failed"] == [failed["run_id"]] and rec["complete"]
    (reports / "suite").mkdir()
    (reports / "suite" / xv.SUMMARY_NAME).write_text(json.dumps(rec))
    monkeypatch.setattr(x3_jobs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(x3_jobs, "tracked_and_unmodified", lambda rel: (True, ""))
    monkeypatch.setattr(x3_jobs, "repo_relative", lambda p: str(p))
    assert len(x3_jobs.new_checkpoints(reports)) == 11
    jobs = x3_jobs.jobs(reports)
    assert len(jobs) == 23 and ("bc_rnn", 4) not in {(j["family"], j["seed"]) for j in jobs}
    # A run the summary does not record as failed stops the gate.
    bad = {**rec, "failed": []}
    (reports / "suite" / xv.SUMMARY_NAME).write_text(json.dumps(bad))
    with pytest.raises(x3_jobs.X3GateError):
        x3_jobs.jobs(reports)
    # A selection that differs from the summary stops the gate.
    bad = json.loads(json.dumps(rec))
    bad["runs"][0]["selection"]["sha256"] = "0" * 64
    (reports / "suite" / xv.SUMMARY_NAME).write_text(json.dumps(bad))
    with pytest.raises(x3_jobs.X3GateError):
        x3_jobs.new_checkpoints(reports)


def test_unresolved_runs_have_no_summary(tmp_path: Path) -> None:
    rows = [{"family": "bc_mlp", "seed": 3, "run_id": "r", "status": "training",
             "selected": False, "missing": [], "manifest": None}]
    with pytest.raises(xv.ValidationError):
        xv.summary_record(rows, tmp_path)
    rows[0].update(status="complete", missing=[5000])
    with pytest.raises(xv.ValidationError):
        xv.summary_record(rows, tmp_path)
