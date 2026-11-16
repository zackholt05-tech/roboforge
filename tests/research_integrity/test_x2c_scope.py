"""Only the X2-c code path reaches the X2 held-out suite, and only through the X2 freeze
(DECISIONS D-123, D-130).

- Outside the X2 suite module and its generator, only the X2-c files name the X2 held-out
  suite, by its literal or by the ``HELDOUT_SUITE`` constant.
- The X2-c evaluator admits only the held-out suite (check mode: only the X2 validation
  suite), takes the suite from the freeze's block, refuses a checkpoint the freeze does not
  list, and refuses all of it before it creates a directory or loads a checkpoint. The X2-a and
  X2-b evaluator still refuses the held-out suite.
- The X2-c scripts load the committed freeze before any held-out step; the container script
  needs the host's blob attestation.
- X2-c records go to ``runs/x2c/`` (never ``runs/x2/``); a directory is never reused; a second
  attempt needs the failed first one; the output scan finds held-out outputs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboforge.paths import REPO_ROOT
from roboforge.replication import x3_states
from roboforge.sim import eval_states as es
from roboforge.x2 import binding
from roboforge.x2 import evaluate as x2eval
from roboforge.x2 import heldout as xh
from roboforge.x2 import heldout_gate as xg
from roboforge.x2 import states as x2s

# Files that may name the X2 held-out suite (literal or constant): the suite module and its
# generator (D-123) and the X2-c files that parse the freeze, build it and scan for outputs.
HELDOUT_NAMERS = {"src/roboforge/x2/states.py", "scripts/generate_x2_states.py",
                  "src/roboforge/x2/heldout.py", "src/roboforge/x2/heldout_gate.py",
                  "src/roboforge/x2/freeze_doc.py", "scripts/write_x2_freeze.py"}


def test_only_x2c_files_name_the_heldout_suite() -> None:
    offenders = []
    for root in ("src", "scripts"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text()
            if (x2s.HELDOUT_SUITE in text or "HELDOUT_SUITE" in text) and rel not in \
                    HELDOUT_NAMERS:
                offenders.append(rel)
    assert offenders == [], f"files naming the X2 held-out suite: {offenders}"


def test_the_x2c_suite_check() -> None:
    xh.check_x2c_suite("can", x2s.HELDOUT_SUITE)
    xh.check_x2c_suite("can", x2s.VALIDATION_SUITE, check=True)
    for suite in (x2s.VALIDATION_SUITE, es.LOCKED_SUITE, "validation", "robustness_base",
                  x3_states.suite_name()):
        with pytest.raises(ValueError):
            xh.check_x2c_suite("can", suite)
    for suite in (x2s.HELDOUT_SUITE, es.LOCKED_SUITE, "validation"):
        with pytest.raises(ValueError):
            xh.check_x2c_suite("can", suite, check=True)
    with pytest.raises(ValueError):
        xh.check_x2c_suite("lift", x2s.HELDOUT_SUITE)
    with pytest.raises(ValueError):  # the X2-a / X2-b evaluator is unchanged
        x2eval.check_x2_eval_suite("can", x2s.HELDOUT_SUITE)


def _fake_freeze(name: str = x2s.HELDOUT_SUITE) -> dict:
    run = {"seed": 0, "train_run_id": "t", "step": 5000, "checkpoint_sha256": "a" * 64,
           "validation_successes": 1}
    fams = {f: {"chosen_trial": 1, "overrides": [], "config_sha256": "c" * 64,
                "runs": [{**run, "seed": s, "train_run_id": f"{f}{s}"} for s in (0, 1, 2)]}
            for f in ("bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet")}
    return {"suite": {"task": "can", "name": name, "n_states": 100,
                      "manifest_content_sha256": "0" * 64},
            "evaluation_platform": {"platform_id": "p"}, "families": fams}


def _train_dir(tmp: Path, name: str = "t") -> Path:
    run = tmp / name
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "run_id": name, "status": "complete", "config": {"task": "can", "family": "bc_mlp",
                                                         "seed": 0},
        "checkpoints": [{"step": 5000, "sha256": "a" * 64, "path": str(run / "missing.pt")}]}))
    return run


@pytest.mark.parametrize("case", ["no_freeze", "check_with_freeze", "unlisted", "suite"])
def test_the_evaluator_refuses_before_touching_anything(case, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(binding, "active", lambda: True)
    runs_dir = tmp_path / "runs"
    kw = {"family": "bc_mlp", "seed": 0, "expected_sha256": "a" * 64, "runs_dir": runs_dir}
    train = _train_dir(tmp_path, "bc_mlp0")
    with pytest.raises((ValueError, xh.X2FreezeError)):
        if case == "no_freeze":
            xh.evaluate_x2c(train, 5000, freeze=None, **kw)
        elif case == "check_with_freeze":
            xh.evaluate_x2c(train, 5000, freeze=_fake_freeze(), check=True, **kw)
        elif case == "unlisted":
            xh.evaluate_x2c(_train_dir(tmp_path, "other"), 5000, freeze=_fake_freeze(),
                            freeze_git_blob="b" * 40, **kw)
        else:
            xh.evaluate_x2c(train, 5000, freeze=_fake_freeze(x2s.VALIDATION_SUITE),
                            freeze_git_blob="b" * 40, **kw)
    assert not runs_dir.exists()


def test_the_evaluator_needs_the_binding(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        xh.evaluate_x2c(_train_dir(tmp_path), 5000, family="bc_mlp", seed=0,
                        expected_sha256="a" * 64, runs_dir=tmp_path / "runs",
                        freeze=_fake_freeze(), freeze_git_blob="b" * 40)


def test_the_held_out_suite_comes_only_from_the_freeze() -> None:
    src = (REPO_ROOT / "src/roboforge/x2/heldout.py").read_text()
    assert 'suite = freeze["suite"]["name"]' in src
    assert "x2s.HELDOUT_SUITE" in src.split("def parse_x2_freeze", 1)[1].split("def ", 1)[0]
    run = (REPO_ROOT / "scripts/x2c_run.py").read_text().split("def main(", 1)[1]
    assert "xg.check_gate(session, decision=a.decision)" in run
    assert run.index("check_gate(") < run.index("run_batch(gate, todo")
    ev = (REPO_ROOT / "scripts/x2c_eval.py").read_text()
    assert "xh.load_x2_freeze(expected_blob=a.freeze_blob)" in ev
    assert 'p.error("a held-out evaluation needs --freeze-blob")' in ev
    assert ev.index("load_x2_freeze(") < ev.index("evaluate_x2c(")
    an = (REPO_ROOT / "scripts/analyze_x2c.py").read_text()
    assert "xh.load_x2_freeze()" in an and "xg.check_inputs(freeze)" in an


def test_records_never_reuse_a_directory_and_never_go_to_runs_x2(tmp_path) -> None:
    assert xh.RUNS_SUBDIR[False] == "x2c" and str(x2eval.RUNS_SUBDIR) == "x2/evals"
    out = xh.prepare_out_dir(tmp_path, "e", check=False)
    assert out == tmp_path / "x2c" / "e"
    with pytest.raises(FileExistsError):
        xh.prepare_out_dir(tmp_path, "e", check=False)
    with pytest.raises(FileNotFoundError):
        xh.prepare_out_dir(tmp_path, "f", check=False, attempt=2)
    assert xh.prepare_out_dir(tmp_path, "e", check=False, attempt=2) == (
        tmp_path / "x2c" / "attempt2" / "e")
    assert xh.prepare_out_dir(tmp_path, "g", check=True) == tmp_path / "x2c_checks" / "g"
    with pytest.raises(ValueError):
        xh.prepare_out_dir(tmp_path, "h", check=True, attempt=2)


def test_the_output_scan_finds_heldout_outputs_anywhere(tmp_path) -> None:
    for d in ("reports", "logs", "runs"):
        (tmp_path / d).mkdir()
    assert xg.existing_heldout_outputs(tmp_path) == []
    eid = f"r__{x2s.HELDOUT_SUITE}_step0005000__p"
    (tmp_path / "runs" / "x2c" / eid).mkdir(parents=True)
    (tmp_path / "logs" / f"{x2s.suite_task()}_{x2s.HELDOUT_SUITE}_0000").mkdir()
    (tmp_path / "reports" / "s.json").write_text(json.dumps({"suite": x2s.HELDOUT_SUITE,
                                                             "episodes": []}))
    (tmp_path / "reports" / "states.json").write_text(json.dumps({"suite": x2s.HELDOUT_SUITE}))
    assert xg.existing_heldout_outputs(tmp_path) == [
        f"logs/{x2s.suite_task()}_{x2s.HELDOUT_SUITE}_0000", "reports/s.json",
        f"runs/x2c/{eid}"]


def test_outputs_of_a_frozen_job_are_its_own() -> None:
    job = {"eval_run_id": "e1"}
    p = xg.job_paths(job, 17)
    found = [p["summary"], p["records"], f"{p['records']}/episodes/x", p["log"],
             "runs/x2c/e2", "reports/session17/x2/heldout/evals/e1.json.bak"]
    assert xg.foreign_outputs(found, [job], 17) == [
        "runs/x2c/e2", "reports/session17/x2/heldout/evals/e1.json.bak"]


def test_isolation_allows_only_the_x2c_tree() -> None:
    snap = {"runs_top_level": {"names": ["a", "x2"]}, "runs_x2": {"sha256": "1"},
            "runs_x3": {"sha256": "2"}, "frozen_level1_heldout": {"sha256": "3"}}
    after = json.loads(json.dumps(snap))
    after["runs_top_level"]["names"].append("x2c")
    assert xg.isolation_problems(snap, after) == []
    after["runs_top_level"]["names"].append("other")
    after["runs_x2"]["sha256"] = "9"
    assert len(xg.isolation_problems(snap, after)) == 2
