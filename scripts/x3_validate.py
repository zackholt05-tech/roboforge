"""X3: validation evaluations and §7 selections of the 12 X3 runs (DECISIONS D-112).

    ROBOFORGE_EVIDENCE_SESSION=14 caffeinate -i uv run --frozen python scripts/x3_validate.py
    ... --dry-run     # print each run's state and what would launch; launch nothing

``make x3-validate``. Separate replication; not a result. It finishes the evaluation work
the frozen ``scripts/run_suite.py`` left when its tuning choice stopped it (KNOWN_ISSUES
KI-062): every checkpoint of every completed X3 run on the 50 Can validation starts
(``scripts/eval_policy.py`` in the canonical container, the suite's own argv and
environment), then the RESEARCH_PROTOCOL §7 selection of each run
(``scripts/select_checkpoint.py``). See ``roboforge.replication.x3_validation``.

- It never trains. A run still training is waited for; a run with neither a final manifest
  nor a live ``train.py`` process stops it for a manual decision.
- At most 4 containers through the host-wide slots (D-069), diffusion first (plan order).
  Every launch requires a clean tree and at least 4 GiB free.
- A failed evaluation, an evaluation directory without an eligible summary, or a summary
  that does not parse stops new launches; running evaluations are waited for. Nothing is
  retried automatically.
- An existing selection record is checked against its sources and never rewritten.
- Events: ``logs/session<N>/suite/x3_validate/events.jsonl``; per-job output beside it;
  status in ``status.json``. When all 12 runs are selected (or failed in training) it writes
  ``reports/session<N>/suite/x3_validation.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, evidence_reports_dir, evidence_session, repo_relative
from roboforge.provenance import git_revision
from roboforge.replication import x3_validation as xv
from roboforge.slots import ContainerSlots, disk_ok, free_gb


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ps_commands() -> list[str]:
    """Full command lines of every process (``-ww``: never truncated)."""
    out = subprocess.run(["ps", "-axww", "-o", "command="], capture_output=True, text=True,
                         check=True).stdout
    return out.splitlines()


class Validator:
    """Scheduling of the X3 validation evaluations and selections."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.a = args
        self.reports: Path = args.reports_dir
        self.reports_rel = repo_relative(self.reports)
        self.logs: Path = args.log_dir
        self.specs = xv.run_specs()
        self.order = {sp["family"]: i for i, sp in enumerate(
            {sp["family"]: sp for sp in self.specs}.values())}
        self.env = {**os.environ, "ROBOFORGE_EVIDENCE_SESSION": str(evidence_session())}
        self.slots = ContainerSlots()
        self.jobs: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []
        self.stopping = False
        self.rows: list[dict[str, Any]] = []

    def event(self, ev: str, /, **kw: Any) -> None:
        """Append one event to ``events.jsonl`` and echo it."""
        rec = {"utc": _now(), "event": ev, **kw}
        if not self.a.dry_run:
            with (self.logs / "events.jsonl").open("a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps(rec, sort_keys=True), flush=True)

    def stop(self, signum: int, _frame: Any) -> None:
        """SIGINT/SIGTERM: no new launches; running evaluations are waited for."""
        self.stopping = True
        self.event("stop_requested", signal=signum)

    def busy(self) -> set[str]:
        return {j["key"] for j in self.jobs}

    def reap(self) -> None:
        for j in list(self.jobs):
            rc = j["proc"].poll()
            if rc is None:
                continue
            self.jobs.remove(j)
            self.slots.release(j["slot"])
            info = {"kind": "eval", "key": j["key"], "returncode": rc,
                    "wall_s": round(time.time() - j["started"], 1)}
            if rc:
                self.failed.append({**info, "fatal": True})
            self.event("finished", **info)

    def launch_eval(self, rid: str, step: int) -> bool:
        """Launch one evaluation if a host-wide container slot is free."""
        if xv.partial_eval_dirs(REPO_ROOT / "runs", rid, step):
            raise xv.ValidationError(f"{rid} step {step}: an evaluation directory exists "
                                     "without an eligible summary (partial or failed)")
        if git_revision()["dirty"]:
            raise xv.ValidationError("working tree is dirty (code/configs/docs): refusing "
                                     "to launch")
        if not disk_ok():
            raise xv.ValidationError(f"only {free_gb():.1f} GiB free: refusing to launch")
        key = f"{rid}@{step}"
        slot = self.slots.acquire(key)
        if slot is None:
            return False
        argv = xv.eval_argv(rid, step, self.reports_rel)
        log = self.logs / "eval" / f"{rid}_step{step:07d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=log.open("a"),
                                stderr=subprocess.STDOUT, env=self.env,
                                start_new_session=True)
        self.slots.attach(slot, proc.pid)
        self.jobs.append({"key": key, "proc": proc, "slot": slot, "started": time.time()})
        self.event("launch", kind="eval", key=key, argv=argv, pid=proc.pid,
                   log=repo_relative(log))
        return True

    def select(self, rid: str) -> None:
        argv = xv.select_argv(sys.executable, rid, self.reports_rel)
        log = self.logs / "select" / f"{rid}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as fh:
            rc = subprocess.run(argv, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                env=self.env, check=False).returncode
        if rc:
            raise xv.ValidationError(f"selection of {rid} failed (exit {rc}); see {log}")
        chosen = json.loads(xv.selection_path(self.reports, rid).read_text())["chosen"]
        self.event("selected", key=rid, step=chosen["step"], successes=chosen["successes"],
                   first_success_step_mean=chosen["first_success_step_mean"])

    def write_status(self, queue: int) -> None:
        if self.a.dry_run:
            return
        st = {"utc": _now(), "stopping": self.stopping, "failed": self.failed,
              "eval_queue": queue,
              "running": [{"key": j["key"], "elapsed_s": round(time.time() - j["started"])}
                          for j in self.jobs],
              "runs": [{k: r[k] for k in ("run_id", "status", "selected")}
                       | {"n_missing": len(r["missing"])} for r in self.rows]}
        (self.logs / "status.json").write_text(json.dumps(st, indent=1) + "\n")

    def tick(self) -> bool:
        """One scheduling step; False when nothing is left to do."""
        self.reap()
        if any(f["fatal"] for f in self.failed):
            self.stopping = True
        self.rows = xv.survey(self.specs, self.reports, REPO_ROOT / "runs", self.busy(),
                              ps_commands())
        lost = [r["run_id"] for r in self.rows if r["status"] == "missing"]
        if lost:
            raise xv.ValidationError(f"no final training manifest and no train.py process: "
                                     f"{lost}")
        busy = self.busy()
        for r in self.rows:
            if (r["status"] == "complete" and not r["missing"] and not r["selected"]
                    and not self.stopping and not self.a.dry_run):
                self.select(r["run_id"])
                r["selected"] = True
        queue = sorted((self.order[r["family"]], r["seed"], s, r["run_id"])
                       for r in self.rows if r["status"] == "complete"
                       for s in r["missing"] if f"{r['run_id']}@{s}" not in busy)
        if self.a.dry_run:
            for r in self.rows:
                print(f"{r['run_id']:45s} {r['status']:9s} to evaluate {len(r['missing']):2d}"
                      f" selected {r['selected']}")
            print(f"{len(queue)} evaluations queued now; nothing launched (dry run)")
            return False
        if not self.stopping:
            for _, _, step, rid in queue:
                if len(self.jobs) >= self.a.max_eval or not self.launch_eval(rid, step):
                    break
        self.write_status(len(queue))
        if self.stopping:
            return bool(self.jobs)
        return not (not self.jobs and xv.resolved(self.rows))

    def finish(self) -> str:
        rec = xv.summary_record(self.rows, self.reports)
        rec["git"], rec["generated_utc"] = git_revision(), _now()
        out = self.reports / "suite" / xv.SUMMARY_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise xv.ValidationError(f"{repo_relative(out)} exists")
        out.write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n")
        return repo_relative(out)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--max-eval", type=int, default=4)
    p.add_argument("--poll-s", type=float, default=15.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.max_eval > 4:
        p.error("Gate C (D-051) allows at most 4 containers")
    a.reports_dir = evidence_reports_dir()
    a.log_dir = REPO_ROOT / "logs" / f"session{evidence_session()}" / "suite" / "x3_validate"
    if not a.dry_run:
        a.log_dir.mkdir(parents=True, exist_ok=True)
    v = Validator(a)
    signal.signal(signal.SIGINT, v.stop)
    signal.signal(signal.SIGTERM, v.stop)
    v.event("start", plan=xv.PLAN, decision=xv.DECISION, git=git_revision(),
            limits={"eval": a.max_eval}, dry_run=a.dry_run)
    code, summary = 0, None
    try:
        while v.tick():
            time.sleep(a.poll_s)
        if not a.dry_run and not v.stopping:
            summary = v.finish()
    except Exception as e:  # noqa: BLE001 - recorded, never silent (KI-036)
        expected = isinstance(e, xv.ValidationError)
        v.event("validation_error", error=str(e) if expected else f"{type(e).__name__}: {e}",
                **({} if expected else {"traceback": traceback.format_exc()}))
        v.stopping, code = True, 2
        while v.jobs:  # never orphan running evaluations
            time.sleep(a.poll_s)
            v.reap()
    if any(f["fatal"] for f in v.failed):
        code = code or 1
    if v.stopping and not code:
        code = 3
    v.write_status(0)
    v.event("end", failed=v.failed, exit=code, summary=summary)
    return code


if __name__ == "__main__":
    sys.exit(main())
