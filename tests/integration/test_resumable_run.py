"""Resumable, interrupt-safe training runs on the real Lift data (Session 5).

An interrupted and resumed run must equal an uninterrupted run of the same config: every
cadence checkpoint content-identical (every tensor bitwise, every other value equal; file
hashes can differ through pickle memoization, see ``payload_differences``), the same metric
rows apart from wall-clock fields, and the same final offline metrics. The tests also cover
what a hard kill leaves behind (an open segment, a torn ``.pt.tmp``, an unindexed checkpoint,
metric rows past the resume point), duplicate-launch refusal, same-launch restarts (Modal
preemption), and real SIGINT/SIGTERM delivery to ``scripts/train.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from roboforge.data.pipeline import load_task_artifacts
from roboforge.data.processed import processed_path
from roboforge.paths import REPO_ROOT
from roboforge.provenance import sha256_file
from roboforge.train.checkpoint import checkpoint_differences
from roboforge.train.config import load_config, run_id
from roboforge.train.journal import ResumeError, RunExistsError, RunJournal
from roboforge.train.run import train_run
from roboforge.train.stop import StopControl

pytestmark = pytest.mark.data
CONFIGS = REPO_ROOT / "configs" / "models"
TINY = ["train.max_steps=60", "train.ckpt_every=20", "train.val_every=20",
        "train.log_every=10"]


@pytest.fixture(scope="module", autouse=True)
def _processed(lift_lowdim_path) -> None:
    m = load_task_artifacts("lift")["task_manifest"]
    if not processed_path("lift", m["processed"]["converter_version"]).exists():
        pytest.skip("processed data not built (run `make build-data`)")


def _cfg(family: str = "bc_mlp", extra: list[str] | None = None):
    return load_config(CONFIGS / f"{family}.yaml", TINY + (extra or []))


def _rows(run_dir: Path) -> list[dict]:
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    for r in rows:
        r.pop("wall_s", None)  # timing differs between segments by construction
    return rows


def _same_checkpoints(m: dict, ref: dict) -> None:
    """Every cadence checkpoint is content-identical (tensors bitwise) to the reference's."""
    assert [c["step"] for c in m["checkpoints"]] == [c["step"] for c in ref["checkpoints"]]
    for c, r in zip(m["checkpoints"], ref["checkpoints"], strict=True):
        assert sha256_file(REPO_ROOT / c["path"]) == c["sha256"]  # the index is truthful
        assert checkpoint_differences(REPO_ROOT / c["path"], REPO_ROOT / r["path"]) == []


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> dict:
    """Uninterrupted reference runs (BC-MLP and diffusion)."""
    out = {}
    for fam in ("bc_mlp", "diffusion_unet"):
        root = tmp_path_factory.mktemp(f"ref_{fam}")
        m = train_run(_cfg(fam), root, "test")
        out[fam] = {"manifest": m, "dir": root / m["run_id"]}
    return out


@pytest.mark.parametrize("family", ["bc_mlp", "diffusion_unet"])
@pytest.mark.parametrize("stop_at", [30, 40])  # off-cadence and on-cadence
def test_interrupted_and_resumed_equals_uninterrupted(tmp_path, reference, family,
                                                       stop_at) -> None:
    cfg = _cfg(family)
    first = train_run(cfg, tmp_path, "test", stop=StopControl(stop_at_step=stop_at))
    assert first["status"] == "interrupted" and first["step"] == stop_at
    run_dir = tmp_path / first["run_id"]
    assert not (run_dir / "manifest.json").exists()
    if stop_at % 20:
        assert first["resume_checkpoint"]["step"] == stop_at
        assert (run_dir / "resume" / f"step_{stop_at:07d}.pt").is_file()
    else:
        assert first["resume_checkpoint"] is None  # the cadence checkpoint suffices
    m = train_run(cfg, tmp_path, "test", resume=True)
    ref = reference[family]["manifest"]
    assert m["status"] == "complete" and m["reload_check"]["passed"]
    assert [c["step"] for c in m["checkpoints"]] == [20, 40, 60]
    _same_checkpoints(m, ref)  # content-identical, tensors bitwise
    assert _rows(run_dir) == _rows(reference[family]["dir"])
    assert m["final_offline"] == ref["final_offline"]
    assert len(m["resume_checkpoints"]) == (1 if stop_at % 20 else 0)
    ends = [e for e in m["segments"] if e["event"] == "end"]
    assert [e["reason"] for e in ends] == ["interrupted", "complete"]
    assert m["timing"]["n_segments"] == 2 and m["timing"]["wall_s"] > 0


def test_hard_kill_leftovers_are_quarantined_and_resume_is_exact(tmp_path, reference) -> None:
    cfg = _cfg("bc_mlp")
    first = train_run(cfg, tmp_path, "test", stop=StopControl(stop_at_step=40))
    run_dir = tmp_path / first["run_id"]
    j = RunJournal(run_dir)
    # What a SIGKILL during step 41..50 can leave: no end record (we drop it), a torn
    # temporary checkpoint, a checkpoint renamed into place without its index line, metric
    # rows past the resume point and a torn last metrics line.
    seg = (run_dir / "segments.jsonl").read_text().splitlines()
    (run_dir / "segments.jsonl").write_text(seg[0] + "\n")
    (run_dir / "checkpoints" / "step_0000060.pt.tmp").write_bytes(b"partial")
    shutil.copy(run_dir / "checkpoints" / "step_0000040.pt",
                run_dir / "checkpoints" / "step_0000060.pt")
    with (run_dir / "metrics.jsonl").open("a") as fh:
        fh.write(json.dumps({"kind": "train", "step": 50, "loss": 9.9}) + "\n{\"kind\": \"tr")
    assert len(j.open_segments()) == 1
    m = train_run(cfg, tmp_path, "test", resume=True, stale_after_s=0)
    assert m["status"] == "complete"
    _same_checkpoints(m, reference["bc_mlp"]["manifest"])
    assert _rows(run_dir) == _rows(reference["bc_mlp"]["dir"])
    start = [e for e in m["segments"] if e["event"] == "start"][1]
    assert sorted(start["orphans"]) == ["checkpoints/step_0000060.pt",
                                        "checkpoints/step_0000060.pt.tmp"]
    assert start["metrics_rows_dropped"] == 2
    assert [e["event"] for e in m["segments"]].count("lost") == 1
    assert (run_dir / "orphans" / "segment001" / "checkpoints" / "step_0000060.pt").is_file()


def test_launch_lock_refuses_duplicates_and_allows_same_launch_restart(tmp_path) -> None:
    cfg = _cfg("bc_mlp")
    first = train_run(cfg, tmp_path, "test", launch_id="L1",
                      stop=StopControl(stop_at_step=30))
    with pytest.raises(RunExistsError):
        train_run(cfg, tmp_path, "test", launch_id="L2")  # a different launch, same run ID
    with pytest.raises(RunExistsError):
        train_run(cfg, tmp_path, "test")  # a fresh nonce is a different launch too
    m = train_run(cfg, tmp_path, "test", launch_id="L1")  # preemption restart: resumes
    assert m["status"] == "complete" and m["launch"]["launch_id"] == "L1"
    again = train_run(cfg, tmp_path, "test", launch_id="L1")  # idempotent after the end
    assert again == m
    with pytest.raises(ResumeError):
        train_run(cfg, tmp_path, "test", resume=True)  # ended runs are immutable
    assert first["run_id"] == m["run_id"] == run_id(cfg, "test")


def test_resume_refusals(tmp_path) -> None:
    cfg = _cfg("bc_mlp")
    with pytest.raises(ResumeError):
        train_run(cfg, tmp_path, "test", resume=True)  # nothing to resume
    first = train_run(cfg, tmp_path, "test", stop=StopControl(stop_at_step=30))
    run_dir = tmp_path / first["run_id"]
    seg = (run_dir / "segments.jsonl").read_text().splitlines()
    (run_dir / "segments.jsonl").write_text(seg[0] + "\n")  # looks like it is still running
    with pytest.raises(ResumeError, match="may still be running"):
        train_run(cfg, tmp_path, "test", resume=True)  # fresh heartbeat, open segment
    stored = json.loads((run_dir / "config.json").read_text())
    stored["train"]["max_steps"] = 999
    (run_dir / "config.json").write_text(json.dumps(stored))
    with pytest.raises(ResumeError, match="config"):
        train_run(cfg, tmp_path, "test", resume=True, stale_after_s=0)
    sidecar = run_dir / "resume.jsonl"
    stored["train"]["max_steps"] = 60
    (run_dir / "config.json").write_text(json.dumps(stored))
    row = json.loads(sidecar.read_text())
    row["sha256"] = "0" * 64
    sidecar.write_text(json.dumps(row) + "\n")
    with pytest.raises(ResumeError, match="SHA-256"):
        train_run(cfg, tmp_path, "test", resume=True, stale_after_s=0)


@pytest.mark.slow
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_real_signal_stops_cleanly_and_resumes(tmp_path, reference, sig) -> None:
    """Deliver a real signal to scripts/train.py; it must save, exit 3 and resume exactly."""
    args = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--config",
            str(CONFIGS / "diffusion_unet.yaml"), "--prefix", "test", "--runs-dir",
            str(tmp_path)] + [a for t in TINY for a in ("--set", t)]
    rid = run_id(_cfg("diffusion_unet"), "test")
    proc = subprocess.Popen(args, cwd=REPO_ROOT, env={**os.environ},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    progress = tmp_path / rid / "progress.json"
    deadline = time.monotonic() + 120
    while not progress.exists() and time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.05)
    assert progress.exists(), proc.communicate()[0]
    proc.send_signal(sig)
    out = proc.communicate(timeout=120)[0]
    assert proc.returncode == 3, out
    end = [json.loads(x) for x in (tmp_path / rid / "segments.jsonl").read_text()
           .splitlines()][-1]
    assert end["reason"] == "interrupted" and end["detail"] == f"signal {sig.name}"
    resumed = subprocess.run(args + ["--resume"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=300)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    m = json.loads((tmp_path / rid / "manifest.json").read_text())
    _same_checkpoints(m, reference["diffusion_unet"]["manifest"])
    assert _rows(tmp_path / rid) == _rows(reference["diffusion_unet"]["dir"])
