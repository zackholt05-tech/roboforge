"""X2-c: run the 12 frozen X2 checkpoints on the X2 held-out starts, once (DECISIONS D-130).

    ROBOFORGE_EVIDENCE_SESSION=17 uv run --frozen python scripts/x2c_run.py --dry-run
    ROBOFORGE_EVIDENCE_SESSION=17 uv run --frozen python scripts/x2c_run.py

**Gate (on the host, before anything launches;** :func:`roboforge.x2.heldout_gate.check_gate`):
a clean tree; ``docs/extensions/X2_FREEZE.md`` committed unmodified; every file it pins and
the Level 1 frozen code unchanged; the preregistration, the X2-b test configuration and the
held-out suite equal to the freeze's pins; the 12 checkpoint files re-hashed; and no X2
held-out output anywhere in ``reports/``, ``logs/`` or ``runs/``.

**Jobs.** The 12 checkpoints of the freeze, diffusion first (longest first). Each runs in its
own canonical container (``scripts/x2c_eval.py``) with the freeze's committed blob ID, at most
4 at a time through the host-wide container slots (D-069), under the D-125 host limits of the
freeze (:class:`roboforge.x2.host.HostGuard`: one launch per fresh 60-s sample, no launch
above 3,000 swap pages/s, below 4 GiB free or on a dirty tree). A dirty tree stops all further
launches.

**No automatic rerun.** A failed job is recorded and not relaunched. ``--decision D-NNN`` (a
heading of the committed DECISIONS.md) allows a later batch: jobs with a summary are skipped,
jobs never launched run as attempt 1, and a job whose first attempt failed before writing its
summary runs once more as attempt 2 (the frozen engineering rule, for a fault outside the
policy). A job that failed twice is never run again.

**Isolation.** ``logs/session<N>/x2/heldout/isolation_{before,after}[_<k>].json``: the
``runs/`` listing and the digests of ``runs/x2``, ``runs/x3`` and the frozen Level 1 held-out
records; after the batch only ``runs/x2c`` may be new, and every held-out output must belong to
a frozen job. Events go to ``events.jsonl``, host samples to ``host.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, evidence_session, repo_relative
from roboforge.provenance import git_revision
from roboforge.slots import ContainerSlots
from roboforge.x2 import heldout_gate as xg
from roboforge.x2.host import HostGuard, now_utc, strip_counters, tree_clean

POLL_S = 10
_STOP = {"requested": False}


def log_dir() -> Path:
    """``logs/session<N>/x2/heldout``."""
    d = REPO_ROOT / "logs" / f"session{evidence_session()}" / "x2" / "heldout"
    d.mkdir(parents=True, exist_ok=True)
    return d


def event(kind: str, **fields: Any) -> None:
    """Append one event to ``events.jsonl`` and echo it."""
    row = {"utc": now_utc(), "event": kind, **fields}
    with (log_dir() / "events.jsonl").open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def write_new(stem: str, obj: dict[str, Any]) -> Path:
    """``<stem>.json``, or ``<stem>_<k>.json`` for a later batch: never overwritten."""
    k, path = 1, log_dir() / f"{stem}.json"
    while path.exists():
        k += 1
        path = log_dir() / f"{stem}_{k}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(obj, indent=1) + "\n")
    return path


def plan(gate: dict[str, Any]) -> list[dict[str, Any]]:
    """The jobs to launch now, each with its attempt (none is ``done`` or failed twice)."""
    out = []
    for j in gate["jobs"]:
        st = gate["status"][j["eval_run_id"]]
        if st == "todo":
            out.append({**j, "attempt": 1})
        elif st == "failed1":
            if gate["decision"] is None:
                raise SystemExit(f"{j['eval_run_id']} failed before; a rerun needs --decision")
            out.append({**j, "attempt": 2})
    return out


def command(j: dict[str, Any], blob: str, decision: str | None) -> list[str]:
    """The container command of one job."""
    cmd = ["infra/containers/run_sim_cpu.sh", "python", "scripts/x2c_eval.py",
           "--train-run", f"runs/x2/{j['train_run_id']}", "--step", str(j["step"]),
           "--family", j["family"], "--seed", str(j["seed"]),
           "--expected-sha256", j["checkpoint_sha256"], "--freeze-blob", blob]
    if j["attempt"] == 2:
        cmd += ["--attempt", "2", "--rerun-decision", str(decision)]
    return cmd


def run_batch(gate: dict[str, Any], todo: list[dict[str, Any]], max_eval: int) -> list[str]:
    """Launch and wait; return the evaluation IDs that failed or were not launched."""
    session, blob = evidence_session(), gate["freeze_blob"]
    guard, pool = HostGuard(gate["freeze"]["limits"]), ContainerSlots()
    running: list[tuple[subprocess.Popen, dict[str, Any], float, Path]] = []
    pending, failed, stop = list(todo), [], False
    while pending or running:
        if guard.due():
            s = guard.sample()
            with (log_dir() / "host.jsonl").open("a") as f:
                f.write(json.dumps(strip_counters(s)) + "\n")
        if _STOP["requested"] and not stop:
            stop = True
            event("stop_requested", pending=[j["eval_run_id"] for j in pending])
        while pending and len(running) < max_eval and not stop:
            if not tree_clean():
                event("dirty_tree_stop")
                stop = True
                break
            why = guard.hold_reason()
            if why is not None:
                if why == "disk" and not running:
                    event("disk_stop")
                    stop = True
                break
            slot = pool.acquire(pending[0]["eval_run_id"])
            if slot is None:
                break  # every host-wide container slot is taken (D-069)
            j = pending.pop(0)
            cmd = command(j, blob, gate["decision"])
            lp = REPO_ROOT / xg.job_paths(j, session)["log"]
            lp.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=lp.open("a"),
                                    stderr=subprocess.STDOUT,
                                    env={**os.environ,
                                         "ROBOFORGE_EVIDENCE_SESSION": str(session)})
            pool.attach(slot, proc.pid)
            guard.launched()
            running.append((proc, j, time.time(), slot))
            event("launch", pid=proc.pid, log=repo_relative(lp), argv=cmd[1:],
                  attempt=j["attempt"], family=j["family"], seed=j["seed"],
                  eval_run_id=j["eval_run_id"])
            break  # one launch per fresh host sample (D-125)
        time.sleep(POLL_S)
        for item in list(running):
            proc, j, t0, slot = item
            if proc.poll() is None:
                continue
            running.remove(item)
            pool.release(slot)
            ok = proc.returncode == 0 and (REPO_ROOT / xg.job_paths(j, session)[
                "summary"]).exists()
            event("finished", eval_run_id=j["eval_run_id"], returncode=proc.returncode,
                  wall_s=round(time.time() - t0, 1), ok=ok, attempt=j["attempt"])
            if not ok:
                failed.append(j["eval_run_id"])
        if stop and not running:
            break
    return failed + [j["eval_run_id"] for j in pending]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--max-eval", type=int, default=4)
    p.add_argument("--decision", help="recorded decision for a later batch or the one rerun")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.max_eval <= 4:
        p.error("1 to 4 containers (Gate C, D-051; D-125)")
    session = evidence_session()
    try:
        gate = xg.check_gate(session, decision=a.decision)
    except xg.X2cGateError as e:
        raise SystemExit(f"X2-c gate: {e}") from e
    todo = plan(gate)
    if a.dry_run:
        for j in gate["jobs"]:
            st = gate["status"][j["eval_run_id"]]
            print(f"{st:8s} {j['family']:15s} seed {j['seed']} step {j['step']:6d} "
                  f"{j['checkpoint_sha256'][:16]} {j['eval_run_id']}")
        print(json.dumps({"jobs": len(gate["jobs"]), "to_launch": len(todo),
                          "freeze_blob": gate["freeze_blob"], "suite": gate["suite"],
                          "content_sha256": gate["content_sha256"],
                          "existing_outputs": len(gate["existing"]), "git": git_revision()}))
        return 0
    signal.signal(signal.SIGINT, lambda *_: _STOP.update(requested=True))
    before = xg.isolation_snapshot()
    pb = write_new("isolation_before", {"utc": now_utc(), "git": git_revision(), **before})
    event("start", n=len(todo), decision=a.decision, git=git_revision(),
          freeze_blob=gate["freeze_blob"], suite=gate["suite"],
          content_sha256=gate["content_sha256"], isolation_before=repo_relative(pb),
          jobs=[{k: j[k] for k in ("family", "seed", "train_run_id", "step",
                                   "checkpoint_sha256", "eval_run_id", "attempt")}
                for j in todo])
    bad = run_batch(gate, todo, a.max_eval)
    after = xg.isolation_snapshot()
    problems = xg.isolation_problems(before, after)
    foreign = xg.foreign_outputs(xg.existing_heldout_outputs(), gate["jobs"], session)
    if foreign:
        problems.append(f"held-out outputs of no frozen job: {foreign[:5]}")
    pa = write_new("isolation_after", {"utc": now_utc(), "git": git_revision(), **after,
                                       "problems": problems,
                                       "isolation": "PASS" if not problems else "FAIL"})
    event("end", failed_or_not_launched=bad, isolation="PASS" if not problems else "FAIL",
          isolation_after=repo_relative(pa))
    return 1 if bad or problems else 0


if __name__ == "__main__":
    sys.exit(main())
