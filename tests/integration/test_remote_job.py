"""The remote training job in a Modal-shaped layout: drill, restart, attempt cap (Session 5).

Gate A showed three things a local run cannot:

* Modal mounts the runs Volume through a symlink (``/app/runs -> /__modal/volumes/<id>``);
* ``modal.exception.simulate_preemption`` raises ``KeyboardInterrupt`` instead of sending
  SIGINT, and re-arms on every restart;
* Modal reschedules an input whose container exits on an interrupt, even with ``retries=0``.

This test runs :func:`roboforge.compute.remote.train_job` itself, in-process, with the
repository root moved to a temporary directory whose ``runs`` is such a symlink. It checks:

* the drill delivers a real SIGINT, which the job's handler turns into a resume checkpoint;
* a restart of the same launch is not re-armed and completes;
* the attempt cap refuses to train again;
* every recorded path is repository-relative (``runs/<run_id>/...``).
"""

from __future__ import annotations

import signal
from pathlib import Path

import pytest

import roboforge.paths
import roboforge.train.run
from roboforge.compute import remote
from roboforge.compute.plan import train_plan
from roboforge.paths import REPO_ROOT
from roboforge.train.journal import RunJournal


@pytest.fixture
def modal_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "app"
    root.mkdir()
    vol = tmp_path / "__modal" / "volumes" / "vo-test"
    vol.mkdir(parents=True)
    (root / "runs").symlink_to(vol, target_is_directory=True)
    monkeypatch.setenv("ROBOFORGE_DATA_ROOT", str(roboforge.paths.data_root()))  # like /data
    monkeypatch.setattr(roboforge.paths, "REPO_ROOT", root)
    monkeypatch.setattr(roboforge.train.run, "REPO_ROOT", root)
    return root


def test_drill_restart_and_attempt_cap_in_a_symlinked_runs_mount(modal_root: Path) -> None:
    plan = train_plan(REPO_ROOT / "configs" / "models" / "bc_mlp.yaml", task="lift", seed=0,
                      prefix="t5remote", timeout_s=600, events=[],
                      overrides=["train.max_steps=1500", "train.ckpt_every=500",
                                 "train.val_every=500", "train.log_every=100"],
                      simulate_preemption_s=2)
    run_dir = modal_root / "runs" / plan["run_id"]

    # Attempt 0: the drill's SIGINT reaches the job's handler; the job writes a resume
    # checkpoint and re-raises, as it must for Modal's preemption restart.
    try:
        with pytest.raises(KeyboardInterrupt, match="signal SIGINT"):
            remote.train_job(plan)
    finally:
        remote._disarm()  # the fired drill keeps its SIGKILL alarm armed
    j = RunJournal(run_dir)
    (start0, end0) = j.segments()
    assert start0["host"]["preemption_drill"]["armed"]
    assert start0["host"]["attempt"] == 0 and start0["host"]["signals_installed"]
    assert end0["reason"] == "interrupted" and end0["detail"] == "signal SIGINT"
    assert end0["resume_checkpoint"]["path"].startswith(f"runs/{plan['run_id']}/")
    assert j.resume_checkpoints()
    assert remote.launch_attempts(run_dir, plan["launch_id"]) == 1
    assert signal.getsignal(signal.SIGALRM) == signal.SIG_DFL

    # The cap refuses another attempt of this launch without training.
    capped = {**plan, "policy": {**plan["policy"], "max_attempts_per_launch": 1}}
    res = remote.train_job(capped)
    assert res["status"] == "error" and "attempt cap" in res["detail"]
    assert len(j.segments()) == 2

    # Attempt 1 (Modal's restart of the same input): not re-armed; it resumes and completes.
    res = remote.train_job(plan)
    assert res["status"] == "complete" and res["attempt"] == 1
    assert res["preemption_drill"] == {"seconds": 2, "armed": False,
                                       "reason": "attempt 1 of this launch: drills arm on "
                                                 "attempt 0 only"}
    starts = [e for e in j.segments() if e["event"] == "start"]
    assert starts[1]["resumed_from"]["kind"] == "resume"
    prefix = f"runs/{plan['run_id']}/"
    assert starts[1]["resumed_from"]["path"].startswith(prefix)
    assert all(c["path"].startswith(prefix) for c in j.checkpoints())
    assert all(c["path"].startswith(prefix) for c in j.resume_checkpoints())
    assert [c["step"] for c in res["detail"]["checkpoints"]] == [500, 1000, 1500]
    assert all(c["path"].startswith(prefix) for c in res["detail"]["checkpoints"])
