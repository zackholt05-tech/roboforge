"""X3: run the 24 evaluations on the X3 held-out starts, once (DECISIONS D-111).

    ROBOFORGE_EVIDENCE_SESSION=14 uv run --frozen python scripts/run_x3.py [--dry-run]

**Separate replication.**

**Jobs.** The 4 families × 6 seeds of Can:
- seeds 0–2: the frozen checkpoints, read from the committed ``docs/FINAL_TEST_FREEZE.md``;
- seeds 3–5: the RESEARCH_PROTOCOL §7 selections of the 12 X3 training runs of
  ``configs/experiments/x3_can.yaml``, read from their committed selection records.

**Gate (all on the host, before anything launches).**
- The working tree is clean.
- ``docs/extensions/X3_PREREGISTRATION.md`` is committed unmodified and names the X3 suite's
  manifest content hash; the suite's data file matches its manifest.
- Every selection record of a new run is committed unmodified, follows the §7 rule, and its
  chosen SHA-256 equals the training manifest's.

**Launch.** Each evaluation runs in its own canonical container (``scripts/eval_x3.py``), at
most 4 at a time, through the host-wide container slots (D-069). The tree must be clean at
every launch, and no launch happens below 4 GiB free.

**Restarts.**
- A job whose summary exists is skipped.
- A job whose record directory exists without a summary is refused and reported. It is never
  overwritten.
- A failed job is recorded and not relaunched.

Events go to ``logs/session<N>/x3/events.jsonl``.
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

from roboforge.paths import REPO_ROOT, evidence_reports_dir, evidence_session, repo_relative
from roboforge.provenance import git_revision
from roboforge.replication.x3 import RUNS_SUBDIR
from roboforge.replication.x3_jobs import X3GateError, check_gate, jobs
from roboforge.slots import ContainerSlots, disk_ok, free_gb


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--max-eval", type=int, default=4)
    p.add_argument("--poll-s", type=float, default=10.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.max_eval > 4:
        p.error("at most 4 containers (Gate C, D-051)")
    reports = evidence_reports_dir()
    try:
        gate = check_gate()
        todo = jobs(reports)
    except X3GateError as e:
        raise SystemExit(f"X3 gate: {e}") from e
    summaries = reports / "x3" / "evals"
    records = REPO_ROOT / "runs" / RUNS_SUBDIR[False]
    logs = REPO_ROOT / "logs" / f"session{evidence_session()}" / "x3"
    status = {}
    for j in todo:
        done = (summaries / f"{j['eval_run_id']}.json").exists()
        partial = (records / j["eval_run_id"]).exists() and not done
        status[j["eval_run_id"]] = "done" if done else ("PARTIAL" if partial else "todo")
    if a.dry_run:
        for j in todo:
            print(f"{status[j['eval_run_id']]:7s} {j['family']:15s} seed {j['seed']} "
                  f"step {j['step']:6d} {j['eval_run_id']}")
        print(f"{len(todo)} jobs; gate {json.dumps(gate)}")
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
    event("start", n=len(pending), git=git_revision(), gate=gate,
          jobs=[{k: j[k] for k in ("family", "seed", "train_run_id", "step",
                                   "checkpoint_sha256", "source", "eval_run_id")}
                for j in todo])
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
            cmd = ["infra/containers/run_sim_cpu.sh", "python", "scripts/eval_x3.py",
                   "--train-run", f"runs/{j['train_run_id']}", "--step", str(j["step"]),
                   "--family", j["family"], "--seed", str(j["seed"]),
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
                  **{k: j[k] for k in ("family", "seed", "eval_run_id")})
        time.sleep(a.poll_s)
        for item in list(running):
            proc, j, t0, slot = item
            if proc.poll() is None:
                continue
            running.remove(item)
            pool.release(slot)
            ok = proc.returncode == 0 and (summaries / f"{j['eval_run_id']}.json").exists()
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
