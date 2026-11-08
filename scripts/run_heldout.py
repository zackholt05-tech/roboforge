"""Run every frozen held-out evaluation once, at most 4 canonical containers at a time.

    ROBOFORGE_EVIDENCE_SESSION=7 uv run --frozen python scripts/run_heldout.py   # make eval-final

Refuses unless the tree is clean, ``docs/FINAL_TEST_FREEZE.md`` is committed unmodified and
the frozen code is unchanged. Launches ``scripts/eval_heldout.py --task T --index i`` in its
own container for every entry of every frozen task (Gate C: <= 4 containers, D-051). An
entry whose summary already exists is skipped (it ran); a failed entry is recorded and never
relaunched automatically (an engineering rerun is a manual, recorded decision, §14.6).
Events: ``logs/session<N>/heldout/events.jsonl``. ``--rehearsal-suite`` + ``--draft-freeze``
rehearse the whole path on a non-locked suite before the freeze.
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

from roboforge.eval import freeze as fz
from roboforge.eval.heldout import eval_id
from roboforge.paths import REPO_ROOT, evidence_reports_dir, evidence_session, repo_relative
from roboforge.provenance import git_revision

PLATFORM = "linux-aarch64-container-1f685638f28c"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--max-eval", type=int, default=4)
    p.add_argument("--rehearsal-suite", default=None)
    p.add_argument("--draft-freeze", type=Path, default=None)
    p.add_argument("--poll-s", type=float, default=10.0)
    a = p.parse_args(argv)
    if a.max_eval > 4:
        p.error("Gate C (D-051) allows at most 4 containers")
    rehearsal = a.rehearsal_suite is not None
    if git_revision()["dirty"]:
        raise SystemExit("working tree is dirty (code/configs/docs): refusing to launch")
    frozen = fz.parse_freeze(a.draft_freeze.read_text()) if rehearsal else fz.load_freeze()
    if not rehearsal and fz.code_problems(frozen):
        raise SystemExit(f"frozen code changed: {fz.code_problems(frozen)}")
    sess = evidence_session()
    logs = REPO_ROOT / "logs" / f"session{sess}" / ("rehearsal/heldout" if rehearsal
                                                    else "heldout")
    reports = logs if rehearsal else evidence_reports_dir("heldout")
    logs.mkdir(parents=True, exist_ok=True)

    def event(ev: str, **kw: object) -> None:
        rec = {"utc": _now(), "event": ev, **kw}
        with (logs / "events.jsonl").open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps(rec, sort_keys=True), flush=True)

    todo = []
    for task in a.tasks or list(frozen["tasks"]):
        for i, e in enumerate(fz.entries(frozen, task)):
            eid = eval_id(e, PLATFORM, rehearsal=rehearsal)
            if (reports / f"{eid}.json").exists():
                event("skip_done", task=task, index=i, eval_run_id=eid)
                continue
            todo.append((task, i, eid))
    event("start", n=len(todo), rehearsal=rehearsal, git=git_revision(),
          freeze=repo_relative(a.draft_freeze) if rehearsal else fz.FREEZE_REL)
    running: list[tuple[subprocess.Popen, str, int, str, float]] = []
    failed = []
    while todo or running:
        while todo and len(running) < a.max_eval:
            task, i, eid = todo.pop(0)
            if git_revision()["dirty"]:
                raise SystemExit("working tree became dirty: refusing to launch more")
            cmd = ["infra/containers/run_sim_cpu.sh", "python", "scripts/eval_heldout.py",
                   "--task", task, "--index", str(i)]
            if rehearsal:
                cmd += ["--rehearsal-suite", a.rehearsal_suite, "--draft-freeze",
                        repo_relative(a.draft_freeze)]
            log = (logs / "jobs" / f"{eid}.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log.open("a"),
                                    stderr=subprocess.STDOUT,
                                    env={**os.environ,
                                         "ROBOFORGE_EVIDENCE_SESSION": str(sess)})
            running.append((proc, task, i, eid, time.time()))
            event("launch", task=task, index=i, eval_run_id=eid, pid=proc.pid,
                  log=repo_relative(log))
        time.sleep(a.poll_s)
        for job in list(running):
            proc, task, i, eid, t0 = job
            if proc.poll() is None:
                continue
            running.remove(job)
            ok = proc.returncode == 0 and (reports / f"{eid}.json").exists()
            event("finished", task=task, index=i, eval_run_id=eid,
                  returncode=proc.returncode, wall_s=round(time.time() - t0, 1), ok=ok)
            if not ok:
                failed.append(eid)
    event("end", failed=failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
