"""X1: run the 45 clamped BC evaluations on the Session 7 robustness sets, once (D-107).

    ROBOFORGE_EVIDENCE_SESSION=13 uv run --frozen python scripts/run_x1_clamp.py [--dry-run]

**Exploratory, after the freeze.**

**Jobs.** The 9 frozen BC checkpoints per task are BC-MLP, BC-RNN and BC-MLP-chunk × seeds 0–2,
read from the two test configuration records. Each record is checked against the SHA-256 that
the Session 7 robustness analysis recorded. Every checkpoint runs on every robustness set of its
task: Lift 9 × 3 + Can 9 × 2 = 45 evaluations.

**Launch.** Each evaluation runs in its own canonical container
(``scripts/eval_x1_clamp.py``), at most 4 at a time, through the host-wide container slots
(D-069). The tree must be clean at every launch.

**Restarts.**
- A job whose summary exists is skipped.
- A job whose record directory exists without a summary is refused and reported. It is never
  overwritten.
- A failed job is recorded and not relaunched.

Events go to ``logs/session<N>/x1_clamp/events.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.exploratory.clamp import RUNS_SUBDIR, x1_eval_run_id
from roboforge.paths import REPO_ROOT, evidence_reports_dir, evidence_session, repo_relative
from roboforge.provenance import git_revision, sha256_file
from roboforge.sim import robustness as rb
from roboforge.slots import ContainerSlots, disk_ok, free_gb

PLATFORM = "linux-aarch64-container-1f685638f28c"
BC_FAMILIES = ("bc_mlp", "bc_rnn", "bc_mlp_chunk")
# The frozen test configuration records, with the SHA-256 that the Session 7 robustness
# analysis recorded (reports/session7/robustness/analysis.json, "configs").
CONFIGS = {
    "lift": ("reports/session6/lift_test_configuration.json",
             "38ac3c293bc5b2cd72b970b2044ed485a14800e7353b07e8956f7231d10961de"),
    "can": ("reports/session7/can_test_configuration.json",
            "116f10a45c243a0cecb6aabc9c48167507f290c70bf08f2f29df6bfc6184f690"),
}


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def bc_checkpoints(task: str) -> list[dict[str, Any]]:
    """The 9 frozen BC checkpoints of ``task``, in family and seed order (hash-checked record)."""
    rel, want = CONFIGS[task]
    path = REPO_ROOT / rel
    got = sha256_file(path)
    if got != want:
        raise SystemExit(f"{rel}: SHA-256 {got} != recorded {want}")
    rec = json.loads(path.read_text())
    if rec["task"] != task:
        raise SystemExit(f"{rel}: task {rec['task']} != {task}")
    return [{"task": task, "family": fam, "seed": s["seed"], "train_run_id": s["train_run_id"],
             "step": int(s["selected_step"]), "checkpoint_sha256": s["checkpoint_sha256"]}
            for fam in BC_FAMILIES for s in rec["families"][fam]["seeds"]]


def jobs() -> list[dict[str, Any]]:
    """All 45 jobs: every BC checkpoint on every robustness set of its task."""
    out = []
    for task in CONFIGS:
        for c in bc_checkpoints(task):
            for suite in rb.sets_for(task):
                out.append({**c, "suite": suite,
                            "eval_run_id": x1_eval_run_id(c["train_run_id"], c["step"], suite,
                                                          PLATFORM, clamp=True)})
    if len({j["eval_run_id"] for j in out}) != len(out):
        raise SystemExit("duplicate jobs")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--max-eval", type=int, default=4)
    p.add_argument("--poll-s", type=float, default=10.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.max_eval > 4:
        p.error("at most 4 containers (Gate C, D-051)")
    todo = jobs()
    reports = evidence_reports_dir("x1_clamp") / "evals"
    records = REPO_ROOT / "runs" / RUNS_SUBDIR[True]
    logs = REPO_ROOT / "logs" / f"session{evidence_session()}" / "x1_clamp"
    status = {}
    for j in todo:
        done = (reports / f"{j['eval_run_id']}.json").exists()
        partial = (records / j["eval_run_id"]).exists() and not done
        status[j["eval_run_id"]] = "done" if done else ("PARTIAL" if partial else "todo")
    if a.dry_run:
        for j in todo:
            print(f"{status[j['eval_run_id']]:7s} {j['eval_run_id']}")
        print(f"{len(todo)} jobs")
        return 0
    bad = [k for k, v in status.items() if v == "PARTIAL"]
    if bad:
        raise SystemExit(f"record directories without a summary (never overwritten): {bad}")
    if git_revision()["dirty"]:
        raise SystemExit("working tree is dirty (code/configs/docs): refusing to launch")
    logs.mkdir(parents=True, exist_ok=True)

    def event(ev: str, **kw: object) -> None:
        rec = {"utc": _now(), "event": ev, **kw}
        with (logs / "events.jsonl").open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps(rec, sort_keys=True), flush=True)

    pending = []
    for j in todo:
        if status[j["eval_run_id"]] == "done":
            event("skip_done", eval_run_id=j["eval_run_id"])
        else:
            pending.append(j)
    event("start", n=len(pending), git=git_revision(),
          configs={k: {"path": v[0], "sha256": v[1]} for k, v in CONFIGS.items()})
    running: list[tuple[subprocess.Popen, dict[str, Any], float, Path]] = []
    failed = []
    pool, stop = ContainerSlots(), False
    while pending or running:
        while pending and len(running) < a.max_eval and not stop:
            if git_revision()["dirty"]:
                raise SystemExit("working tree became dirty: refusing to launch more")
            if not disk_ok():
                event("low_disk", free_gb=round(free_gb(), 2))
                stop = True
                break
            slot = pool.acquire(pending[0]["eval_run_id"])
            if slot is None:
                break  # every host-wide container slot is taken (D-069)
            j = pending.pop(0)
            cmd = ["infra/containers/run_sim_cpu.sh", "python", "scripts/eval_x1_clamp.py",
                   "--train-run", f"runs/{j['train_run_id']}", "--step", str(j["step"]),
                   "--suite", j["suite"], "--family", j["family"],
                   "--expected-sha256", j["checkpoint_sha256"]]
            log = logs / "jobs" / f"{j['eval_run_id']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log.open("a"),
                                    stderr=subprocess.STDOUT,
                                    env={**os.environ,
                                         "ROBOFORGE_EVIDENCE_SESSION": str(evidence_session())})
            pool.attach(slot, proc.pid)
            running.append((proc, j, time.time(), slot))
            event("launch", pid=proc.pid, log=repo_relative(log), argv=cmd[1:],
                  **{k: j[k] for k in ("task", "family", "seed", "suite", "eval_run_id")})
        time.sleep(a.poll_s)
        for item in list(running):
            proc, j, t0, slot = item
            if proc.poll() is None:
                continue
            running.remove(item)
            pool.release(slot)
            ok = proc.returncode == 0 and (reports / f"{j['eval_run_id']}.json").exists()
            event("finished", eval_run_id=j["eval_run_id"], returncode=proc.returncode,
                  wall_s=round(time.time() - t0, 1), ok=ok)
            if not ok:
                failed.append(j["eval_run_id"])
        if stop and not running:
            break
    event("end", failed=failed, not_launched=[j["eval_run_id"] for j in pending])
    return 1 if failed or pending else 0


if __name__ == "__main__":
    sys.exit(main())
